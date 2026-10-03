using CUDA
using CUDA.CUSOLVER
#using CUDA.CUSOLVER: heevd
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics
using Dates
using Serialization
# Cuda.jl only supports real symmetric matrix to get eigenvalue

const REAL_T = Float32
const COMPLEX_T = ComplexF32
const CONVERGENCE_TOL = 1f-5
# Keep the original tolerance above for the preliminary mean-field and
# Hermiticity checks.  The constrained-field inversion uses a deliberately
# separate global L2 tolerance so this diagnostic change is unambiguous.
const FIELD_CONVERGENCE_TOL = 1f-4
const AMPLITUDE_MAX = 0.5f0
const FIELD_ALPHA_MIN = 0.5f0
const FIELD_ALPHA_MAX = 64f0
const FIELD_ALPHA_RELATIVE_BAND = 0.02f0
const FIELD_ALPHA_SHRINK_PATIENCE = 2
const FIELD_ALPHA_SECANT_INTERVAL = 5
const FIELD_ALPHA_MIN_ALIGNMENT = 0.5f0
const FIELD_ALPHA_LOG_BLEND = 0.5f0
const FIELD_ALPHA_MAX_GROWTH = 1.5f0
const FIELD_ALPHA_MAX_SHRINK = 0.5f0
const FIELD_MAX_ITERATIONS = 3000
const FAILURE_TOP_SITES = 20
const SAVE_FULL_FAILURE_HISTORY = lowercase(get(ENV,
    "SDW_SAVE_FULL_FIELD_HISTORY","true")) in ("1","true","yes")


function init_Neel_kernel!(nSt, dim, fil, Δ, mx, my, mz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        j = (idx - 1) ÷  dim + 1
        k = (idx - 1) %  dim + 1
        mz[idx] = Δ * (mod(j + k, 2) - 0.5f0) * 2f0
        # 1 0 1 0 1 0 when fil = 0.5, Δ = 0.5
    end
    return nothing
end

function init_sdw_hmn_kernel!(nSt, spin_dof, tnn, hmn, nnLu, nnRu, nnTu, nnBu, Bx,By,Bz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        hmn[spin_dof*idx-1, nnRu[idx]]  = -tnn # idxu,Ru
        hmn[spin_dof*idx, nnRu[idx]+1] = -tnn # idxd,Rd

        hmn[spin_dof*idx-1, nnTu[idx]] = -tnn # idxu,Tu
        hmn[spin_dof*idx, nnTu[idx]+1] = -tnn # idxd,Td

        #symmetry
        hmn[nnRu[idx], spin_dof*idx-1]  = -tnn # Ru,idxu
        hmn[nnRu[idx]+1, spin_dof*idx] = -tnn # Rd,idxd

        hmn[nnTu[idx], spin_dof*idx-1] = -tnn # Tu,idxu
        hmn[nnTu[idx]+1, spin_dof*idx] = -tnn # Td,idxd

        #Interaction Hamiltonian
        # -B*s = -B * 0.5 * quadratic electron operators
        hmn[spin_dof*idx-1, spin_dof*idx]   = -0.5f0*(Bx[idx]-im*By[idx]) #
        hmn[spin_dof*idx,   spin_dof*idx-1] = -0.5f0*(Bx[idx]+im*By[idx])
        hmn[spin_dof*idx-1, spin_dof*idx-1] = -0.5f0*Bz[idx] 
        hmn[spin_dof*idx,   spin_dof*idx]   = 0.5f0*Bz[idx] 

        #hmn[idx, idx] = vnn * (nn[nnL[idx]] + nn[nnR[idx]] + nn[nn [idx]] + nn[nnB[idx]]) + vList[idx]
    end
    return nothing
end

function updt_sdw_hmn_kernel!(nSt, spin_dof, tnn, hmn, nnLu, nnRu, nnTu, nnBu, Bx,By,Bz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx<=nSt
        hmn[spin_dof*idx-1, nnRu[idx]]  = -tnn # idxu,Ru
        hmn[spin_dof*idx, nnRu[idx]+1] = -tnn # idxd,Rd

        hmn[spin_dof*idx-1, nnTu[idx]] = -tnn # idxu,Tu
        hmn[spin_dof*idx, nnTu[idx]+1] = -tnn # idxd,Td

        #symmetry
        hmn[nnRu[idx], spin_dof*idx-1]  = -tnn # Ru,idxu
        hmn[nnRu[idx]+1, spin_dof*idx] = -tnn # Rd,idxd

        hmn[nnTu[idx], spin_dof*idx-1] = -tnn # Tu,idxu
        hmn[nnTu[idx]+1, spin_dof*idx] = -tnn # Td,idxd

        #Interaction Hamiltonian
        # -B*s = -B * 0.5 * quadratic electron operators
        hmn[spin_dof*idx-1, spin_dof*idx]   = -0.5f0*(Bx[idx]-im*By[idx]) #
        hmn[spin_dof*idx,   spin_dof*idx-1] = -0.5f0*(Bx[idx]+im*By[idx])
        hmn[spin_dof*idx-1, spin_dof*idx-1] = -0.5f0*Bz[idx] 
        hmn[spin_dof*idx,   spin_dof*idx]   = 0.5f0*Bz[idx] 

        #hmn[idx, idx] = vnn * (nn[nnL[idx]] + nn[nnR[idx]] + nn[nn [idx]] + nn[nnB[idx]]) + vList[idx]
    end
    return nothing
end

function init_sdw_hmn_self_consistent_kernel!(nSt, dim, spin_dof, tnn, hmn, nnLu, nnRu, nnTu, nnBu, mx,my,mz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #m2 = sum(mx.^2 + my.^2 + mz.^2)
    if idx<=nSt
        hmn[spin_dof*idx-1, nnRu[idx]]  = -tnn # idxu,Ru
        hmn[spin_dof*idx, nnRu[idx]+1] = -tnn # idxd,Rd

        hmn[spin_dof*idx-1, nnTu[idx]] = -tnn # idxu,Tu
        hmn[spin_dof*idx, nnTu[idx]+1] = -tnn # idxd,Td

        #symmetry
        hmn[nnRu[idx], spin_dof*idx-1]  = -tnn # Ru,idxu
        hmn[nnRu[idx]+1, spin_dof*idx] = -tnn # Rd,idxd

        hmn[nnTu[idx], spin_dof*idx-1] = -tnn # Tu,idxu
        hmn[nnTu[idx]+1, spin_dof*idx] = -tnn # Td,idxd

        #Interaction Hamiltonian
        #Front -1 comes from hamiltonian
        #m2 is added only to diagonal part
        hmn[spin_dof*idx-1, spin_dof*idx]   = -1*(mx[idx]-im*my[idx]) #
        hmn[spin_dof*idx,   spin_dof*idx-1] = -1*(mx[idx]+im*my[idx])
        hmn[spin_dof*idx-1, spin_dof*idx-1] = -1*mz[idx] 
        hmn[spin_dof*idx,   spin_dof*idx]   = -1*-1*mz[idx] 

        #hmn[idx, idx] = vnn * (nn[nnL[idx]] + nn[nnR[idx]] + nn[nn [idx]] + nn[nnB[idx]]) + vList[idx]
    end
    return nothing
end

#Same with init_sdw code
function updt_sdw_hmn_self_consistent_kernel!(nSt, tnn, hmn, nnLu, nnRu, nnTu, nnBu, mx, my, mz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #m2 = sum(mx.^2 + my.^2 + mz.^2)
    if idx<=nSt

        i_up = 2*idx - 1
        i_dn = 2*idx

        # Neighboring indices (assume already computed and safe)
        j_up_R = nnRu[idx]
        j_dn_R = nnRu[idx] + 1

        j_up_T = nnTu[idx]
        j_dn_T = nnTu[idx] + 1

        hmn[i_up, j_up_R] = -tnn
        hmn[j_up_R, i_up] = -tnn

        hmn[i_dn, j_dn_R] = -tnn
        hmn[j_dn_R, i_dn] = -tnn

        hmn[i_up, j_up_T] = -tnn
        hmn[j_up_T, i_up] = -tnn

        hmn[i_dn, j_dn_T] = -tnn
        hmn[j_dn_T, i_dn] = -tnn

        # Interaction Hamiltonian
        # Front -1 comes from hamiltonian
        # m2 is added only to diagonal part
         # Interaction terms
         hmn[i_up, i_dn] = -1f0 * (mx[idx] - im * my[idx])
         hmn[i_dn, i_up] = -1f0 * (mx[idx] + im * my[idx])
 
         hmn[i_up, i_up] = -1f0 * mz[idx]
         hmn[i_dn, i_dn] = mz[idx]
    end
    return nothing
end


function comp_fermi_kernel!(nSt, μ, kT, energy, fermiF)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # 2*nSt = 2 * number of lattice
    if idx <= 2*nSt
        ratio = (energy[idx] - μ)/kT
        if ratio > 30f0
            ## fermiFactor ≈ 1E-14
            fermiF[idx] = 0f0
        elseif ratio < -30f0
            ## 1 - fermiFactor ≈ 1E-14
            fermiF[idx] = 1f0
        else
            fermiF[idx] = 1f0 / (exp(ratio) + 1f0)
        end
    end
    return nothing
end

# Adjust the chemical potential after every diagonalization so that the
# electron number, rather than the chemical potential, is fixed.
function find_mu_cuda!(nSt, vals, kT::T, target_Ne::T, work;
                       tol::T=T(1e-6), nT_mu::Int=512) where {T<:AbstractFloat}
    kT > zero(T) || error("find_mu_cuda! requires kT > 0")
    μ_low  = minimum(vals) - T(30)*kT
    μ_high = maximum(vals) + T(30)*kT
    nB_mu = cld(length(vals), nT_mu)

    # Bisection is monotone because sum(f_FD) increases with chemical potential.
    while μ_high - μ_low > tol
        μ_mid = (μ_low + μ_high)/T(2)
        @cuda threads=nT_mu blocks=nB_mu comp_fermi_kernel!(
            nSt, μ_mid, kT, vals, work)
        if sum(work) > target_Ne
            μ_high = μ_mid
        else
            μ_low = μ_mid
        end
    end
    return (μ_low + μ_high)/T(2)
end

function comp_m_kernel!(nSt, gs, fermiF, mx, my, mz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # mx,my,mz are N dimensional vector, not 2N
    # Hamiltonian Real elements and symmetric matrix --> eigenvectors are always real
    #gs_conj = conj.(gs)
    if idx <= nSt
        #density = 0.0
        ud = zero(eltype(gs))
        du = zero(eltype(gs))
        uu = zero(eltype(fermiF))
        dd = zero(eltype(fermiF))
        for i in 1:2*nSt
            du += conj(gs[2*idx, i]) * gs[2*idx-1, i] * fermiF[i]
            ud += conj(gs[2*idx-1, i]) * gs[2*idx, i] * fermiF[i]
            dd += abs2(gs[2*idx, i]) * fermiF[i]
            uu += abs2(gs[2*idx-1, i]) * fermiF[i]
        end
        #println(du)
        #println(ud)
        mx[idx] = real(0.5f0*(ud + du))
        my[idx] = real(0.5f0*im*(du - ud))
        mz[idx] = 0.5f0*(uu-dd)
    end
    return nothing
end

function init_hmn_interaction_kernel!(nSt, dim, spin_dof, tnn, hmn, vnn, vList, nn, nnLuu, nnRuu, nnTuu, nnBuu, mx, my, mz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #m2 = mx.^2 + my.^2 + mz.^2
    if idx <= nSt
        #Filling 2Nx2N Hamiltonian
        hmn[spin_dof*idx-1, spin_dof*idx]   = mx[idx]-im*my[idx]
        hmn[spin_dof*idx,   spin_dof*idx-1] = mx[idx]+im*my[idx]
        hmn[spin_dof*idx-1, spin_dof*idx-1] = mz[idx] #+ m2
        hmn[spin_dof*idx,   spin_dof*idx]   = -mz[idx]# + m2
        #hmn[idx, idx] = vnn * (nn[nnL[idx]] + nn[nnR[idx]] + nn[nnT[idx]] + nn[nnB[idx]]) + vList[idx]
    end
    return nothing
end



function computeNN_kernel!(dim, nSt, nnLu, nnRu, nnBu, nnTu, spin_dof)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx<=nSt
        j = (idx - 1) ÷  dim + 1
        k = (idx - 1) %  dim + 1
        jdx = (j - 1) * dim
        bdx = mod(j - 2, dim) * dim
        tdx = mod(j    , dim) * dim
        ldx = mod(k - 2, dim) + 1
        rdx = mod(k    , dim) + 1
        #Find adjacent index upspin index
        nnLu[idx] = spin_dof*(jdx + ldx)-1
        nnRu[idx] = spin_dof*(jdx + rdx)-1
        nnTu[idx] = spin_dof*(bdx + k)-1
        nnBu[idx] = spin_dof*(tdx + k)-1
    end
    return nothing
end

function computeNN_cuda(nT, nB, dim, nSt,spin_dof)
    nnLu = CUDA.zeros(Int64, nSt)
    nnRu = CUDA.zeros(Int64, nSt)
    nnTu = CUDA.zeros(Int64, nSt)
    nnBu = CUDA.zeros(Int64, nSt)
    @cuda threads = nT blocks = nB (
    computeNN_kernel!(dim, nSt, nnLu, nnRu, nnBu, nnTu, spin_dof))
    #println(nnTuu[1])
    return nnLu,nnRu,nnBu,nnTu
end

function comp_Lagrange_kernel!(nSt, Bx,By,Bz, nnL, nnR, nnB, nnT, lnn_x, lnn_y,lnn_z,mx,my,mz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        #lnn[idx]  = potential[idx] - vnn * (nn[nnL[idx]] + nn[nnR[idx]] + nn[nnT[idx]] + nn[nnB[idx]]) - vList[idx]
        lnn_x[idx] = 2*mx[idx] - Bx[idx]
        lnn_y[idx] = 2*my[idx] - By[idx]
        lnn_z[idx] = 2*mz[idx] - Bz[idx]
    end
    return nothing
end

function initialize_random_magnetization(nSt, tnn, kT, μ,spin_dof,k)

    hmn   = CUDA.zeros(COMPLEX_T, (2*nSt, 2*nSt))
    mx       = CUDA.zeros(REAL_T, nSt)
    my       = CUDA.zeros(REAL_T, nSt)
    mz       = CUDA.zeros(REAL_T, nSt)
    fermiF= CUDA.zeros(REAL_T, 2*nSt)

    #Bx,By,Bz = (2*rand(Float64, nSt).-1)*k, (2*rand(Float64, nSt).-1)*k, (2*rand(Float64, nSt).-1)*k
    Bx,By,Bz = (2f0*CUDA.rand(REAL_T, nSt).-1f0)*k, (2f0*CUDA.rand(REAL_T, nSt).-1f0)*k, (2f0*CUDA.rand(REAL_T, nSt).-1f0)*k
    #Bx,By,Bz = zeros(Float64, nSt), zeros(Float64, nSt), ((-1) .^ (0:nSt-1))*k
    #println(k)
    #println(Bx[1])
    #Bx,By,Bz = zeros(Float64, nSt), zeros(Float64, nSt), 0.1*ones(Float64, nSt)
    #Bx,By,Bz = ones(Float64, nSt), ones(Float64, nSt), zeros(Float64, nSt)
    #Bx,By,Bz = k*ones(Float64, nSt), k*ones(Float64, nSt), k*ones(Float64, nSt)
    #Bx,By,Bz = (-1) .^ (0:nSt-1), ones(Float64, nSt), ones(Float64, nSt)
    #Bx,By,Bz = zeros(Float64, nSt), zeros(Float64, nSt), 0.1*ones(Float64, nSt)
    # Generate checkerboard: (i + j) mod 2 gives 0 or 1
    #staggered = [k*(-1) ^ (i + j) for i in 0:16-1, j in 0:16-1]
    #Bx = zeros(Float64, nSt)
    #By = zeros(Float64, nSt)
    #Bz = zeros(Float64, nSt)
    # Flatten to match 1D vector format
    #Bx .= reshape(staggered, nSt)
    #By .= reshape(staggered, nSt)
    #Bz .= reshape(staggered, nSt)

    @cuda threads = nT blocks = nB init_sdw_hmn_kernel!(nSt, spin_dof, tnn, hmn, nnLu, nnRu, nnTu, nnBu, Bx,By,Bz)

    #hmn_cpu = Matrix(hmn)
    hmn_copy = copy(hmn)
    vals, vecs = CUDA.CUSOLVER.heevd!('V', 'U', hmn_copy)
    μ = find_mu_cuda!(nSt, vals, kT, REAL_T(2*nSt)*REAL_T(0.5), fermiF)
    #vals = CuArray(vals)
    #vecs = CuArray(vecs)
    @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, vals, fermiF)
    @cuda threads = nT blocks = nB comp_m_kernel!(nSt, vecs, fermiF, mx, my, mz)
    # @cuda threads = nT blocks = nB comp_bond_kernel!(nSt, vecs, fermiF, nnR, nnT, bondR, bondT)

    # return nn,bondR,bondT
    return mx, my, mz
end



function selfConsistent(dim, nSt, spin_dof, kT, fil, tnn,Δ)

    hmn      = CUDA.zeros(COMPLEX_T, ((2*nSt, 2*nSt)))
    #Define mean-field
    mx       = CUDA.zeros(REAL_T, nSt)
    my       = CUDA.zeros(REAL_T, nSt)
    mz       = CUDA.zeros(REAL_T, nSt)
    fermiF   = CUDA.zeros(REAL_T, 2*nSt)

    @cuda threads = nT blocks = nB init_Neel_kernel!(nSt, dim, fil, Δ, mx, my, mz)
    @cuda threads = nT blocks = nB init_sdw_hmn_self_consistent_kernel!(nSt, dim, spin_dof, tnn, hmn, nnLu, nnRu, nnTu, nnBu, mx, my, mz) #just initialize no loop

    #hmn_cpu = Matrix(hmn)
    GC.gc()
    CUDA.reclaim()
    CUDA.@allowscalar println("Hermitian check: ", all(abs.(Array(hmn .- hmn')) .< CONVERGENCE_TOL))
    #workspace_size = CUDA.CUSOLVER.syevd_buffer_size('V', 'U', hmn)
    #println("Workspace size needed: ", workspace_size * sizeof(ComplexF64) / 1024^2, " MiB")
    hmn_copy = copy(hmn)
    valsInit, vecsInit = CUDA.CUSOLVER.heevd!('V', 'U', hmn_copy)
    println("FIrst is done")
    #valsInit = CuArray(valsInit)
    #vecsInit = CuArray(vecsInit)
    ## define the chemical potential around the filling level
    μ = find_mu_cuda!(nSt, valsInit, kT, REAL_T(2*nSt)*fil, fermiF)
    ## compute the fermi factor
    @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, valsInit, fermiF)
    @cuda threads = nT blocks = nB comp_m_kernel!(nSt, vecsInit, fermiF, mx, my, mz)
    #change "m" based on the current eigenvectors
    for itr in 1:5000
        println("$itr is done")
        #GC.gc()
        #CUDA.reclaim()
        CUDA.@allowscalar println("Hermitian check: ", all(abs.(Array(hmn .- hmn')) .< CONVERGENCE_TOL))
        @cuda threads = nT blocks = nB updt_sdw_hmn_self_consistent_kernel!(nSt, tnn, hmn, nnLu, nnRu, nnTu, nnBu,mx,my,mz)
        #hmn_cpu = Matrix(hmn)
        hmn_copy = copy(hmn)
        vals, vecs = CUDA.CUSOLVER.heevd!('V', 'U', hmn_copy)
        #vals = CuArray(vals)
        #vecs = CuArray(vecs)
        μ = find_mu_cuda!(nSt, vals, kT, REAL_T(2*nSt)*fil, fermiF)
        @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, vals, fermiF)
        @cuda threads = nT blocks = nB comp_m_kernel!(nSt, vecs, fermiF, mx, my, mz)
        Δnew = reduce(+, map(abs, mz)) / REAL_T(nSt)
        println("Δnew", Δnew)
        #println("Δnew",Δnew)
        if abs(Δnew-Δ) < CONVERGENCE_TOL
            Δ = Δnew
            println("step: $(itr), Δ converged at $(Δ), μ: $(μ)")
            break
        else
            Δ = Δnew
        end
    end
    return μ,Δ
end


function comp_energy(nSt, gs, fermiF, t, spin_dof, Bx, By, Bz, nnRu, nnTu)
    # idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # mx,my,mz are N dimensional vector, not 2N
    # Hamiltonian Real elements and symmetric matrix --> eigenvectors are always real
    global total_energy = 0
    gs_conj = conj.(gs)
    kinE = zeros(COMPLEX_T, nSt)
    intE = zeros(COMPLEX_T, nSt)
    sx       = zeros(REAL_T, nSt)
    sy       = zeros(REAL_T, nSt)
    sz       = zeros(REAL_T, nSt)
    

    for idx in 1:nSt
        #density = 0.0
        ud = zero(COMPLEX_T)
        du = zero(COMPLEX_T)
        uu = zero(COMPLEX_T)
        dd = zero(COMPLEX_T)
        kinuu = zero(COMPLEX_T)
        kindd = zero(COMPLEX_T)
        for i in 1:2*nSt
            kinuu += (gs_conj[2*idx-1, i] * gs[nnRu[idx], i] * fermiF[i])
            kinuu += (gs_conj[2*idx-1, i] * gs[nnTu[idx], i] * fermiF[i])
            #kinuu += -t*(gs[nnRu[idx], i] * gs_conj[2*idx-1, i] * fermiF[i])
            #kinuu += -t*(gs[nnTu[idx], i] * gs_conj[2*idx-1, i] * fermiF[i])

            kindd += (gs_conj[2*idx, i] * gs[nnRu[idx]+1, i] * fermiF[i])
            kindd += (gs_conj[2*idx, i] * gs[nnTu[idx]+1, i] * fermiF[i])
            #kindd += (gs[nnRu[idx]+1, i] * gs_conj[2*idx, i] * fermiF[i])
            #kindd += (gs[nnTu[idx]+1, i] * gs_conj[2*idx, i] * fermiF[i])

            du += gs_conj[2*idx, i] * gs[2*idx-1, i] * fermiF[i]
            ud += gs_conj[2*idx-1, i] * gs[2*idx, i] * fermiF[i]
            dd += gs_conj[2*idx, i] * gs[2*idx, i] * fermiF[i]
            uu += gs_conj[2*idx-1, i] * gs[2*idx-1, i] * fermiF[i]
        end
        sx[idx] = 0.5f0*(ud + du)
        sy[idx] = 0.5f0*im*(du - ud)
        sz[idx] = 0.5f0*(uu-dd)
        kinE[idx] = -tnn*(kinuu + kindd)
        intE[idx] = -1*(sx[idx]*Bx[idx] + sy[idx]*By[idx] + sz[idx]*Bz[idx]) 
    end
    total_energy = sum(kinE + intE)
    println("kinetic energy : ",sum(kinE))
    println("interaction energy : ",sum(intE))
    println(total_energy)
    return total_energy
end


function initialize_site_alpha_kernel!(nSt,temp_mx,temp_my,temp_mz,
    target_mx,target_my,target_mz,alpha_site,old_site_residual,
    improvement_streak,worsening_streak,alpha_increases,alpha_decreases,alpha0)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx<=nSt
        rx=temp_mx[idx]-target_mx[idx]
        ry=temp_my[idx]-target_my[idx]
        rz=temp_mz[idx]-target_mz[idx]
        alpha_site[idx]=alpha0
        old_site_residual[idx]=sqrt(rx*rx+ry*ry+rz*rz)
        improvement_streak[idx]=0
        worsening_streak[idx]=0
        alpha_increases[idx]=0
        alpha_decreases[idx]=0
    end
    return nothing
end

function site_field_delta_kernel!(nSt,temp_mx,temp_my,temp_mz,
    target_mx,target_my,target_mz,alpha_site,delta_x,delta_y,delta_z)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx<=nSt
        alpha=alpha_site[idx]
        delta_x[idx]=alpha*(temp_mx[idx]-target_mx[idx])
        delta_y[idx]=alpha*(temp_my[idx]-target_my[idx])
        delta_z[idx]=alpha*(temp_mz[idx]-target_mz[idx])
    end
    return nothing
end

function record_and_adapt_site_alpha_kernel!(nSt,itr,temp_mx,temp_my,temp_mz,
    target_mx,target_my,target_mz,Bx,By,Bz,alpha_site,old_site_residual,
    improvement_streak,worsening_streak,alpha_increases,alpha_decreases,
    hist_mx,hist_my,hist_mz,hist_lx,hist_ly,hist_lz,hist_alpha,
    hist_alpha_candidate,hist_alignment,
    alpha_min,alpha_max,relative_band,shrink_patience,
    secant_interval,min_alignment,log_blend,max_growth,max_shrink)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx<=nSt
        T=eltype(temp_mx)
        cx,cy,cz=temp_mx[idx],temp_my[idx],temp_mz[idx]
        rx=cx-target_mx[idx]; ry=cy-target_my[idx]; rz=cz-target_mz[idx]
        new_residual=sqrt(rx*rx+ry*ry+rz*rz)
        alpha_used=alpha_site[idx]

        # lambda_i and alpha_i used at this exact field iterate.
        lx=T(2)*cx-Bx[idx]; ly=T(2)*cy-By[idx]; lz=T(2)*cz-Bz[idx]
        hist_mx[idx,itr]=cx; hist_my[idx,itr]=cy; hist_mz[idx,itr]=cz
        hist_lx[idx,itr]=lx; hist_ly[idx,itr]=ly; hist_lz[idx,itr]=lz
        hist_alpha[idx,itr]=alpha_used

        # Local Barzilai-Borwein/secant inverse susceptibility. With
        # s=Delta B and y=Delta<m>, alpha=s.s/(s.y) estimates chi^{-1}.
        candidate=T(NaN); control_candidate=T(NaN)
        alignment=T(NaN); valid_secant=false
        if itr>1
            pmx=hist_mx[idx,itr-1]; pmy=hist_my[idx,itr-1]
            pmz=hist_mz[idx,itr-1]
            pBx=T(2)*pmx-hist_lx[idx,itr-1]
            pBy=T(2)*pmy-hist_ly[idx,itr-1]
            pBz=T(2)*pmz-hist_lz[idx,itr-1]
            sx=Bx[idx]-pBx; sy=By[idx]-pBy; sz=Bz[idx]-pBz
            yx=cx-pmx; yy=cy-pmy; yz=cz-pmz
            s2=sx*sx+sy*sy+sz*sz
            y2=yx*yx+yy*yy+yz*yz
            sdoty=sx*yx+sy*yy+sz*yz
            # s2 and y2 are squared Float32 increments and can legitimately
            # be far below eps(Float32). An eps test here incorrectly disables
            # the secant precisely in the low-susceptibility saturation regime.
            # Positivity, finiteness, and directional alignment are the proper
            # scale-free safeguards; candidate clipping controls its magnitude.
            if s2>zero(T) && y2>zero(T) && sdoty>zero(T) &&
               isfinite(s2) && isfinite(y2) && isfinite(sdoty)
                alignment=sdoty/sqrt(s2*y2)
                if isfinite(alignment) && alignment>=min_alignment
                    candidate=s2/sdoty
                    if isfinite(candidate) && candidate>zero(T)
                        control_candidate=clamp(candidate,alpha_min,alpha_max)
                        valid_secant=true
                    end
                end
            end
        end
        hist_alpha_candidate[idx,itr]=candidate
        hist_alignment[idx,itr]=alignment

        old_residual=old_site_residual[idx]
        meaningful_worsening=new_residual>(one(T)+relative_band)*old_residual
        if meaningful_worsening
            worsening_streak[idx]+=1
        else
            worsening_streak[idx]=0
        end

        new_alpha=alpha_used
        if worsening_streak[idx]>=shrink_patience
            new_alpha=max(max_shrink*alpha_used,alpha_min)
            worsening_streak[idx]=0
        elseif valid_secant && itr%secant_interval==0
            # Geometric blending is stable across orders of magnitude. Limit
            # each accepted secant change even if the raw inverse response is huge.
            blended=exp((one(T)-log_blend)*log(alpha_used)+
                log_blend*log(control_candidate))
            lo=max(max_shrink*alpha_used,alpha_min)
            hi=min(max_growth*alpha_used,alpha_max)
            new_alpha=clamp(blended,lo,hi)
        end
        if new_alpha>alpha_used
            alpha_increases[idx]+=1
        elseif new_alpha<alpha_used
            alpha_decreases[idx]+=1
        end
        alpha_site[idx]=new_alpha
        old_site_residual[idx]=new_residual
    end
    return nothing
end

function save_failed_field_history!(stage,iterations,target_mx,target_my,target_mz,
    temp_mx,temp_my,temp_mz,Bx,By,Bz,lnn_x,lnn_y,lnn_z,history)
    stage_tag=replace(stage,r"[^A-Za-z0-9_-]"=>"_")
    prefix=joinpath(experimental_path,"field_failure_$(stage_tag)")

    tx,ty,tz=Array(target_mx),Array(target_my),Array(target_mz)
    cx,cy,cz=Array(temp_mx),Array(temp_my),Array(temp_mz)
    bx,by,bz=Array(Bx),Array(By),Array(Bz)
    lx,ly,lz=Array(lnn_x),Array(lnn_y),Array(lnn_z)
    alpha_final=Array(history.alpha_site)
    increase_count=Array(history.alpha_increases)
    decrease_count=Array(history.alpha_decreases)
    candidate_final=Array(@view history.hist_alpha_candidate[:,iterations])
    alignment_final=Array(@view history.hist_alignment[:,iterations])
    site_residual=sqrt.((cx.-tx).^2 .+(cy.-ty).^2 .+(cz.-tz).^2)
    order=sortperm(site_residual;rev=true)
    top_sites=order[1:min(FAILURE_TOP_SITES,length(order))]

    summary_path=prefix*"_sites.csv"
    open(summary_path,"w") do io
        println(io,"residual_rank,site,target_mx,target_my,target_mz,target_M," *
            "current_mx,current_my,current_mz,current_M,residual_x,residual_y,residual_z," *
            "site_residual,Bx,By,Bz,lambda_x,lambda_y,lambda_z,lambda_M," *
            "lambda_parallel_target,lambda_parallel_current," *
            "alpha_next,raw_secant_alpha_candidate,secant_alignment," *
            "alpha_increases,alpha_decreases")
        for (rank,site) in enumerate(order)
            target_M=sqrt(tx[site]^2+ty[site]^2+tz[site]^2)
            current_M=sqrt(cx[site]^2+cy[site]^2+cz[site]^2)
            lambda_M=sqrt(lx[site]^2+ly[site]^2+lz[site]^2)
            lambda_parallel_target=target_M>eps(REAL_T) ?
                (lx[site]*tx[site]+ly[site]*ty[site]+lz[site]*tz[site])/target_M : REAL_T(NaN)
            lambda_parallel_current=current_M>eps(REAL_T) ?
                (lx[site]*cx[site]+ly[site]*cy[site]+lz[site]*cz[site])/current_M : REAL_T(NaN)
            println(io,join((rank,site,tx[site],ty[site],tz[site],target_M,
                cx[site],cy[site],cz[site],current_M,cx[site]-tx[site],
                cy[site]-ty[site],cz[site]-tz[site],site_residual[site],
                bx[site],by[site],bz[site],lx[site],ly[site],lz[site],lambda_M,
                lambda_parallel_target,lambda_parallel_current,
                alpha_final[site],candidate_final[site],alignment_final[site],
                increase_count[site],decrease_count[site]),','))
        end
    end

    # Copy the full site history only after a failed solve. During normal
    # solves it remains on the GPU and is simply overwritten by the next call.
    hmx=Array(@view history.hist_mx[:,1:iterations])
    hmy=Array(@view history.hist_my[:,1:iterations])
    hmz=Array(@view history.hist_mz[:,1:iterations])
    hlx=Array(@view history.hist_lx[:,1:iterations])
    hly=Array(@view history.hist_ly[:,1:iterations])
    hlz=Array(@view history.hist_lz[:,1:iterations])
    halpha=Array(@view history.hist_alpha[:,1:iterations])
    hcand=Array(@view history.hist_alpha_candidate[:,1:iterations])
    halign=Array(@view history.hist_alignment[:,1:iterations])
    full_history_path=nothing
    if SAVE_FULL_FAILURE_HISTORY
        full_history_path=prefix*"_full_history.jls"
        open(full_history_path,"w") do io
            serialize(io,(
                stage=stage,iterations=iterations,target_mx=tx,target_my=ty,target_mz=tz,
                current_mx=hmx,current_my=hmy,current_mz=hmz,
                lambda_x=hlx,lambda_y=hly,lambda_z=hlz,alpha=halpha,
                raw_secant_alpha_candidate=hcand,secant_alignment=halign))
        end
    end

    trajectory_path=prefix*"_top$(length(top_sites))_trajectories.csv"
    open(trajectory_path,"w") do io
        println(io,"field_iteration,residual_rank,site,target_M,current_M," *
            "current_mx,current_my,current_mz,residual_x,residual_y,residual_z," *
            "site_residual,Bx,By,Bz,lambda_x,lambda_y,lambda_z,lambda_M," *
            "lambda_parallel_target,lambda_parallel_current,alpha," *
            "raw_secant_alpha_candidate,secant_alignment")
        for itr in 1:iterations, (rank,site) in enumerate(top_sites)
            current_M=sqrt(hmx[site,itr]^2+hmy[site,itr]^2+hmz[site,itr]^2)
            rx=hmx[site,itr]-tx[site]; ry=hmy[site,itr]-ty[site]
            rz=hmz[site,itr]-tz[site]
            local_residual=sqrt(rx^2+ry^2+rz^2)
            lambda_M=sqrt(hlx[site,itr]^2+hly[site,itr]^2+hlz[site,itr]^2)
            target_M=sqrt(tx[site]^2+ty[site]^2+tz[site]^2)
            lambda_parallel_target=target_M>eps(REAL_T) ?
                (hlx[site,itr]*tx[site]+hly[site,itr]*ty[site]+
                 hlz[site,itr]*tz[site])/target_M : REAL_T(NaN)
            lambda_parallel_current=current_M>eps(REAL_T) ?
                (hlx[site,itr]*hmx[site,itr]+hly[site,itr]*hmy[site,itr]+
                 hlz[site,itr]*hmz[site,itr])/current_M : REAL_T(NaN)
            hist_Bx=2hmx[site,itr]-hlx[site,itr]
            hist_By=2hmy[site,itr]-hly[site,itr]
            hist_Bz=2hmz[site,itr]-hlz[site,itr]
            println(io,join((itr,rank,site,target_M,current_M,
                hmx[site,itr],hmy[site,itr],hmz[site,itr],rx,ry,rz,
                local_residual,hist_Bx,hist_By,hist_Bz,
                hlx[site,itr],hly[site,itr],hlz[site,itr],lambda_M,
                lambda_parallel_target,lambda_parallel_current,
                halpha[site,itr],hcand[site,itr],halign[site,itr]),','))
        end
    end
    return (summary=summary_path,trajectories=trajectory_path,
        full_history=full_history_path,worst_site=first(order),
        worst_site_residual=site_residual[first(order)])
end

function find_local_field!(dim,hmn,fermiF,nSt,fil,kT,tnn,stp,
    target_mx,target_my,target_mz,mu,k,Bx,By,Bz,
    lnn_x,lnn_y,lnn_z,temp_mx,temp_my,temp_mz;
    stage="unspecified",history)
    @cuda threads=nT blocks=nB init_sdw_hmn_kernel!(nSt,spin_dof,tnn,hmn,
        nnLu,nnRu,nnTu,nnBu,Bx,By,Bz)
    hmn_copy=copy(hmn)
    valsInit,vecsInit=CUDA.CUSOLVER.heevd!('V','U',hmn_copy)
    mu=find_mu_cuda!(nSt,valsInit,kT,REAL_T(2*nSt)*fil,fermiF)
    @cuda threads=nT blocks=nB comp_fermi_kernel!(nSt,mu,kT,valsInit,fermiF)
    @cuda threads=nT blocks=nB comp_m_kernel!(nSt,vecsInit,fermiF,
        temp_mx,temp_my,temp_mz)

    alpha0=clamp(REAL_T(stp),FIELD_ALPHA_MIN,FIELD_ALPHA_MAX)
    @cuda threads=nT blocks=nB initialize_site_alpha_kernel!(nSt,
        temp_mx,temp_my,temp_mz,target_mx,target_my,target_mz,
        history.alpha_site,history.old_site_residual,
        history.improvement_streak,history.worsening_streak,
        history.alpha_increases,history.alpha_decreases,alpha0)
    @cuda threads=nT blocks=nB site_field_delta_kernel!(nSt,
        temp_mx,temp_my,temp_mz,target_mx,target_my,target_mz,
        history.alpha_site,history.delta_x,history.delta_y,history.delta_z)

    converged=false
    iterations=0
    newDiff_x=sqrt(sum(abs2,temp_mx.-target_mx))
    newDiff_y=sqrt(sum(abs2,temp_my.-target_my))
    newDiff_z=sqrt(sum(abs2,temp_mz.-target_mz))
    for itr in 1:FIELD_MAX_ITERATIONS
        iterations=itr
        Bx.-=history.delta_x; By.-=history.delta_y; Bz.-=history.delta_z
        @cuda threads=nT blocks=nB updt_sdw_hmn_kernel!(nSt,spin_dof,tnn,hmn,
            nnLu,nnRu,nnTu,nnBu,Bx,By,Bz)
        hmn_copy=copy(hmn)
        vals,vecs=CUDA.CUSOLVER.heevd!('V','U',hmn_copy)
        mu=find_mu_cuda!(nSt,vals,kT,REAL_T(2*nSt)*fil,fermiF)
        @cuda threads=nT blocks=nB comp_fermi_kernel!(nSt,mu,kT,vals,fermiF)
        @cuda threads=nT blocks=nB comp_m_kernel!(nSt,vecs,fermiF,
            temp_mx,temp_my,temp_mz)

        newDiff_x=sqrt(sum(abs2,temp_mx.-target_mx))
        newDiff_y=sqrt(sum(abs2,temp_my.-target_my))
        newDiff_z=sqrt(sum(abs2,temp_mz.-target_mz))
        @cuda threads=nT blocks=nB record_and_adapt_site_alpha_kernel!(nSt,itr,
            temp_mx,temp_my,temp_mz,target_mx,target_my,target_mz,Bx,By,Bz,
            history.alpha_site,history.old_site_residual,
            history.improvement_streak,history.worsening_streak,
            history.alpha_increases,history.alpha_decreases,
            history.hist_mx,history.hist_my,history.hist_mz,
            history.hist_lx,history.hist_ly,history.hist_lz,history.hist_alpha,
            history.hist_alpha_candidate,history.hist_alignment,
            FIELD_ALPHA_MIN,FIELD_ALPHA_MAX,FIELD_ALPHA_RELATIVE_BAND,
            FIELD_ALPHA_SHRINK_PATIENCE,FIELD_ALPHA_SECANT_INTERVAL,
            FIELD_ALPHA_MIN_ALIGNMENT,FIELD_ALPHA_LOG_BLEND,
            FIELD_ALPHA_MAX_GROWTH,FIELD_ALPHA_MAX_SHRINK)

        if itr==1 || itr%500==0
            println("[field-progress] stage=",stage," iteration=",itr,
                " residuals=",(newDiff_x,newDiff_y,newDiff_z),
                " alpha(min,mean,max)=",(minimum(history.alpha_site),
                sum(history.alpha_site)/REAL_T(nSt),maximum(history.alpha_site)))
        end
        if newDiff_x<FIELD_CONVERGENCE_TOL &&
           newDiff_y<FIELD_CONVERGENCE_TOL &&
           newDiff_z<FIELD_CONVERGENCE_TOL
            converged=true
            break
        end
        @cuda threads=nT blocks=nB site_field_delta_kernel!(nSt,
            temp_mx,temp_my,temp_mz,target_mx,target_my,target_mz,
            history.alpha_site,history.delta_x,history.delta_y,history.delta_z)
    end

    @cuda threads=nT blocks=nB comp_Lagrange_kernel!(nSt,Bx,By,Bz,
        nnLu,nnRu,nnBu,nnTu,lnn_x,lnn_y,lnn_z,temp_mx,temp_my,temp_mz)
    alpha_min=minimum(history.alpha_site)
    alpha_mean=sum(history.alpha_site)/REAL_T(nSt)
    alpha_max=maximum(history.alpha_site)
    failure_report=nothing
    if converged
        println("[field] stage=",stage," iterations=",iterations,
            " residuals=",(newDiff_x,newDiff_y,newDiff_z),
            " alpha(min,mean,max)=",(alpha_min,alpha_mean,alpha_max),
            " tolerance=",FIELD_CONVERGENCE_TOL)
    else
        failure_report=save_failed_field_history!(stage,iterations,
            target_mx,target_my,target_mz,temp_mx,temp_my,temp_mz,
            Bx,By,Bz,lnn_x,lnn_y,lnn_z,history)
        println("[field-nonconverged] stage=",stage," reached ",iterations,
            " iterations; continuing from final field iterate. residuals=",
            (newDiff_x,newDiff_y,newDiff_z)," alpha(min,mean,max)=",
            (alpha_min,alpha_mean,alpha_max)," worst_site=",
            failure_report.worst_site," worst_site_residual=",
            failure_report.worst_site_residual," reports=",failure_report)
    end
    return (converged=converged,iterations=iterations,
        residual_x=newDiff_x,residual_y=newDiff_y,residual_z=newDiff_z,
        final_alpha=alpha_mean,min_alpha=alpha_min,max_alpha=alpha_max,
        alpha_increases=sum(history.alpha_increases),
        alpha_decreases=sum(history.alpha_decreases),
        failure_report=failure_report)
end

dim   = 16
nSt   = dim * dim
# Initial value : half-filling
fil   = 0.5f0
kT    = parse(REAL_T, get(ENV, "SDW_TELEC", "0.01"))
kT_fluc = parse(REAL_T, get(ENV, "SDW_TFLUC", string(kT)))
kT > 0f0 || error("SDW_TELEC must be positive")
kT_fluc >= 0f0 || error("SDW_TFLUC must be nonnegative")
tnn   = 0.25f0 # t/U
## step for convergence of single-Slater-determinant state
stp   = 5f-1
## step for evolution of order parameter dynamics
dt    = 5f-2
nT    = 512
nB    = cld(nSt, nT)

# Set these equal to recover the isotropic mobility used in your latest code.
# Keep them distinct when amplitude and orientation relax on different scales.
Γpar  = 0.1f0
Γperp = 0.3f0
k = 1f-2
#Random.seed!(parse(Int, get(ARGS, 1, "1")))
CUDA.seed!(2025)
Δ   = 0.05f0 # inital strength of |m0|

# Create the output directory before the preliminary self-consistent solve.
# This leaves an immediate run record even if initialization fails or takes a
# long time before the first dynamical snapshot is written.
base_path = get(ENV, "SDW_OUTPUT_DIR", joinpath(@__DIR__, "outputs"))
mkpath(base_path)
method_tag = "direct_M_reflected_L$(dim)_Telec$(kT)_Tfluc$(kT_fluc)_secant_alpha_continue_v2_$(FIELD_MAX_ITERATIONS)_a0_$(stp)_amin_$(FIELD_ALPHA_MIN)_amax_$(FIELD_ALPHA_MAX)_dt_$(dt)"
run_tag = Dates.format(now(), "yyyymmdd_HHMMSS")
experimental_path = joinpath(base_path, "experimental", method_tag, run_tag)
mkpath(experimental_path)
open(joinpath(experimental_path, "run_status.txt"), "w") do io
    println(io, "status=starting_preliminary_self_consistent_solve")
    println(io, "started_at=$(Dates.format(now(), dateformat\"yyyy-mm-ddTHH:MM:SS\"))")
    println(io, "lattice_dimension=$dim")
    println(io, "number_of_sites=$nSt")
    println(io, "electronic_temperature=$kT")
    println(io, "fluctuation_temperature=$kT_fluc")
end
println("Experimental output directory: ", experimental_path)

println("Hi")
spin_dof = 2
# neighborhood site index
nnLu,nnRu,nnBu,nnTu = computeNN_cuda(nT, nB, dim, nSt, spin_dof) 
println("Before SC loop")
μ,Δ = selfConsistent(dim, nSt, spin_dof, kT, fil, tnn,Δ)
println("Converged chemical potential: ", μ, " converged order parameter: ", Δ)
println("Electronic smearing kT = ", kT, "; fluctuation temperature kT_fluc = ", kT_fluc)
println("Amplitude method: direct reflected M-Heun on [0, ",AMPLITUDE_MAX,
    "]; per-site safeguarded secant alpha starts at ",stp,
    " with (min,max,secant_interval,min_alignment,log_blend,max_growth,max_shrink)=",
    (FIELD_ALPHA_MIN,FIELD_ALPHA_MAX,FIELD_ALPHA_SECANT_INTERVAL,
     FIELD_ALPHA_MIN_ALIGNMENT,FIELD_ALPHA_LOG_BLEND,
     FIELD_ALPHA_MAX_GROWTH,FIELD_ALPHA_MAX_SHRINK))

open(joinpath(experimental_path, "settings.txt"), "w") do io
    println(io, "method=direct reflected M-Heun + original two-stage SIB + per-site safeguarded secant inverse-susceptibility alpha v2 (scale-free secant gate) + continue after field nonconvergence")
    println(io, "lattice_dimension=$dim")
    println(io, "number_of_sites=$nSt")
    println(io, "M_interval=[0,$AMPLITUDE_MAX]")
    println(io, "field_alpha_initial=$stp")
    println(io, "field_alpha_min=$FIELD_ALPHA_MIN")
    println(io, "field_alpha_max=$FIELD_ALPHA_MAX")
    println(io, "field_alpha_relative_band=$FIELD_ALPHA_RELATIVE_BAND")
    println(io, "field_alpha_shrink_patience=$FIELD_ALPHA_SHRINK_PATIENCE")
    println(io, "field_alpha_secant_interval=$FIELD_ALPHA_SECANT_INTERVAL")
    println(io, "field_alpha_min_alignment=$FIELD_ALPHA_MIN_ALIGNMENT")
    println(io, "field_alpha_log_blend=$FIELD_ALPHA_LOG_BLEND")
    println(io, "field_alpha_max_growth=$FIELD_ALPHA_MAX_GROWTH")
    println(io, "field_alpha_max_shrink=$FIELD_ALPHA_MAX_SHRINK")
    println(io, "field_max_iterations=$FIELD_MAX_ITERATIONS")
    println(io, "failure_top_sites=$FAILURE_TOP_SITES")
    println(io, "save_full_failure_history=$SAVE_FULL_FAILURE_HISTORY")
    println(io, "dt=$dt")
    println(io, "Gamma_parallel=$Γpar")
    println(io, "Gamma_perpendicular=$Γperp")
    println(io, "electronic_temperature=$kT")
    println(io, "fluctuation_temperature=$kT_fluc")
    println(io, "longitudinal_noise_std_per_step=$(sqrt(2f0*kT_fluc*Γpar*dt))")
    println(io, "field_tolerance=$FIELD_CONVERGENCE_TOL")
end
open(joinpath(experimental_path, "run_status.txt"), "a") do io
    println(io, "status=preliminary_self_consistent_solve_complete")
    println(io, "initial_chemical_potential=$μ")
    println(io, "initial_order_parameter=$Δ")
end

target_mx, target_my, target_mz = initialize_random_magnetization(nSt, tnn, kT, μ, spin_dof,k)

#init_Neel!(nSt, dim, fil, 0.3, target_mx, target_my, target_mz)
#println("target_mx", target_mx)
#println("target_my", target_my)
#println("target_mz", target_mz)

# Direct reflected-amplitude Langevin dynamics on 0 <= M <= 1/2.
# The continuous model is the normally reflected SDE
#   lambda_parallel_i = e_i dot lambda_i,
#   lambda_i = 2*m_i - B_i,
#   dM_i = Γparallel*lambda_parallel_i*dt
#          + sqrt(2*kT_fluc*Γparallel)*dW_i
#          + dK_lower_i - dK_upper_i.
# There is no additional 2*kT_fluc*Γparallel/M_i geometric drift in this
# experimental flat-M reflected model.
# The finite-step scheme uses a symmetry (mirror) map at both boundaries.
# This is a discretization of the reflected SDE, so dt convergence must be
# checked. The orientation is advanced by the original two-stage SIB method.

@inline function reflect_interval(value, upper)
    T=typeof(value)
    y=mod(value,T(2)*upper)
    return y<=upper ? y : T(2)*upper-y
end

@inline function cayley_rotate(ex,ey,ez,qx,qy,qz)
    T=typeof(ex); half=T(0.5); two=T(2)
    wx,wy,wz=half*qx,half*qy,half*qz
    w2=wx*wx+wy*wy+wz*wz
    edotw=ex*wx+ey*wy+ez*wz
    cx=ey*wz-ez*wy; cy=ez*wx-ex*wz; cz=ex*wy-ey*wx
    den=one(T)+w2; common=one(T)-w2
    return ((common*ex+two*cx+two*edotw*wx)/den,
            (common*ey+two*cy+two*edotw*wy)/den,
            (common*ez+two*cz+two*edotw*wz)/den)
end

function decompose_m_kernel!(nSt,mx,my,mz,M,ex,ey,ez,epsm,Mmax)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx<=nSt
        T=eltype(mx); x,y,z=mx[idx],my[idx],mz[idx]
        mag=sqrt(x*x+y*y+z*z)
        M[idx]=(isfinite(mag)&&mag<=Mmax+T(2e-6)) ? min(mag,Mmax) : T(NaN)
        if mag>epsm
            invmag=one(T)/mag
            ex[idx],ey[idx],ez[idx]=x*invmag,y*invmag,z*invmag
        else
            ex[idx],ey[idx],ez[idx]=zero(T),zero(T),one(T)
        end
    end
    return nothing
end

function mheun_sib_predictor_kernel!(nSt,dt,Γpar,Γperp,kT_fluc,
    epsm,Mmax,M,ex,ey,ez,bx,by,bz,ξpar,ξx,ξy,ξz,
    Mp,Mrawp,epx,epy,epz,mxp,myp,mzp)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx<=nSt
        T=eltype(M); two=T(2)
        mag=M[idx]; ux,uy,uz=ex[idx],ey[idx],ez[idx]
        fx,fy,fz=bx[idx],by[idx],bz[idx]
        lambda_par=ux*fx+uy*fy+uz*fz
        raw=mag+dt*Γpar*lambda_par+
            sqrt(max(two*kT_fluc*Γpar*dt,zero(T)))*ξpar[idx]
        magp=reflect_interval(raw,Mmax)

        if mag>epsm
            ebx=uy*fz-uz*fy; eby=uz*fx-ux*fz; ebz=ux*fy-uy*fx
            ewx=uy*ξz[idx]-uz*ξy[idx]
            ewy=uz*ξx[idx]-ux*ξz[idx]
            ewz=ux*ξy[idx]-uy*ξx[idx]
            gamma=Γperp/mag
            sigma_perp=sqrt(max(two*kT_fluc*Γperp*dt,zero(T)))/mag
            px,py,pz=cayley_rotate(ux,uy,uz,
                dt*(fx-gamma*ebx)-sigma_perp*ewx,
                dt*(fy-gamma*eby)-sigma_perp*ewy,
                dt*(fz-gamma*ebz)-sigma_perp*ewz)
        else
            px,py,pz=ux,uy,uz
        end
        Mrawp[idx]=raw; Mp[idx]=magp
        epx[idx],epy[idx],epz[idx]=px,py,pz
        mxp[idx],myp[idx],mzp[idx]=magp*px,magp*py,magp*pz
    end
    return nothing
end

function midpoint_m_kernel!(nSt,M,ex,ey,ez,Mp,epx,epy,epz,mxs,mys,mzs)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx<=nSt
        T=eltype(M); half=T(0.5)
        magh=half*(M[idx]+Mp[idx])
        mxs[idx]=magh*half*(ex[idx]+epx[idx])
        mys[idx]=magh*half*(ey[idx]+epy[idx])
        mzs[idx]=magh*half*(ez[idx]+epz[idx])
    end
    return nothing
end

function mheun_sib_corrector_kernel!(nSt,dt,Γpar,Γperp,kT_fluc,
    epsm,Mmax,M,ex,ey,ez,bx,by,bz,Mp,epx,epy,epz,bxp,byp,bzp,
    bxs,bys,bzs,ξpar,ξx,ξy,ξz,Mrawnew,mx,my,mz)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx<=nSt
        T=eltype(M); half=T(0.5); two=T(2)
        mag=M[idx]; magp=Mp[idx]
        ux,uy,uz=ex[idx],ey[idx],ez[idx]
        px,py,pz=epx[idx],epy[idx],epz[idx]
        lambda0=ux*bx[idx]+uy*by[idx]+uz*bz[idx]
        lambdap=px*bxp[idx]+py*byp[idx]+pz*bzp[idx]
        raw=mag+half*dt*Γpar*(lambda0+lambdap)+
            sqrt(max(two*kT_fluc*Γpar*dt,zero(T)))*ξpar[idx]
        magnew=reflect_interval(raw,Mmax)

        ehx,ehy,ehz=half*(ux+px),half*(uy+py),half*(uz+pz)
        magh=half*(mag+magp); fx,fy,fz=bxs[idx],bys[idx],bzs[idx]
        if magh>epsm
            ebx=ehy*fz-ehz*fy; eby=ehz*fx-ehx*fz; ebz=ehx*fy-ehy*fx
            ewx=ehy*ξz[idx]-ehz*ξy[idx]
            ewy=ehz*ξx[idx]-ehx*ξz[idx]
            ewz=ehx*ξy[idx]-ehy*ξx[idx]
            gamma=Γperp/magh
            sigma_perp=sqrt(max(two*kT_fluc*Γperp*dt,zero(T)))/magh
            nx,ny,nz=cayley_rotate(ux,uy,uz,
                dt*(fx-gamma*ebx)-sigma_perp*ewx,
                dt*(fy-gamma*eby)-sigma_perp*ewy,
                dt*(fz-gamma*ebz)-sigma_perp*ewz)
        else
            nx,ny,nz=ux,uy,uz
        end
        Mrawnew[idx]=raw; M[idx]=magnew
        mx[idx],my[idx],mz[idx]=magnew*nx,magnew*ny,magnew*nz
    end
    return nothing
end

function glsd_mheun_sib_reflected_step!(mx,my,mz,dt,Γpar,Γperp,
    kT_elec,kT_fluc,bx,by,bz,Bfield_x,Bfield_y,Bfield_z,
    M,ex,ey,ez,Mp,Mrawp,epx,epy,epz,mxp,myp,mzp,
    bxp,byp,bzp,Bxp,Byp,Bzp,mxs,mys,mzs,bxs,bys,bzs,Bxs,Bys,Bzs,
    ξpar,ξx,ξy,ξz,Mrawnew,
    hmn,fermiF,temp_mx_cp,temp_my_cp,temp_mz_cp,field_history,step_id)
    T=eltype(mx); n=length(mx); epsm=T(1e-7); Mmax=T(AMPLITUDE_MAX)
    dt>zero(T)||error("dt must be positive")
    kT_fluc>=zero(T)||error("fluctuation temperature must be nonnegative")
    Γpar>=zero(T)||error("Γparallel must be nonnegative")
    Γperp>=zero(T)||error("Γperp must be nonnegative")
    ξpar .= CUDA.randn(T,n); ξx .= CUDA.randn(T,n)
    ξy .= CUDA.randn(T,n); ξz .= CUDA.randn(T,n)

    @cuda threads=nT blocks=nB decompose_m_kernel!(
        nSt,mx,my,mz,M,ex,ey,ez,epsm,Mmax)
    any(.!isfinite.(M))&&error("invalid input amplitude outside [0,1/2]")
    old_mean_M=sum(M)/T(n); old_max_M=maximum(M); old_min_M=minimum(M)

    @cuda threads=nT blocks=nB mheun_sib_predictor_kernel!(nSt,dt,Γpar,
        Γperp,kT_fluc,epsm,Mmax,M,ex,ey,ez,bx,by,bz,
        ξpar,ξx,ξy,ξz,Mp,Mrawp,epx,epy,epz,mxp,myp,mzp)
    any(.!isfinite.(mxp))&&error("reflected M-Heun predictor became nonfinite")
    pred_lower_reflections=sum(Mrawp .< zero(T))
    pred_upper_reflections=sum(Mrawp .> Mmax)
    pred_raw_min=minimum(Mrawp); pred_raw_max=maximum(Mrawp)

    copyto!(Bxp,Bfield_x); copyto!(Byp,Bfield_y); copyto!(Bzp,Bfield_z)
    endpoint_info=find_local_field!(dim,hmn,fermiF,nSt,fil,kT_elec,tnn,stp,
        mxp,myp,mzp,μ,k,Bxp,Byp,Bzp,bxp,byp,bzp,
        temp_mx_cp,temp_my_cp,temp_mz_cp;
        stage="Heun endpoint, step $step_id",history=field_history)
    @cuda threads=nT blocks=nB midpoint_m_kernel!(nSt,M,ex,ey,ez,Mp,epx,epy,epz,
        mxs,mys,mzs)
    Bxs .= T(0.5).*(Bfield_x.+Bxp); Bys .= T(0.5).*(Bfield_y.+Byp)
    Bzs .= T(0.5).*(Bfield_z.+Bzp)
    midpoint_info=find_local_field!(dim,hmn,fermiF,nSt,fil,kT_elec,tnn,stp,
        mxs,mys,mzs,μ,k,Bxs,Bys,Bzs,bxs,bys,bzs,
        temp_mx_cp,temp_my_cp,temp_mz_cp;
        stage="SIB midpoint, step $step_id",history=field_history)

    @cuda threads=nT blocks=nB mheun_sib_corrector_kernel!(nSt,dt,Γpar,
        Γperp,kT_fluc,epsm,Mmax,M,ex,ey,ez,bx,by,bz,Mp,epx,epy,epz,
        bxp,byp,bzp,bxs,bys,bzs,ξpar,ξx,ξy,ξz,Mrawnew,mx,my,mz)
    (any(.!isfinite.(mx))||any(.!isfinite.(my))||any(.!isfinite.(mz)))&&
        error("reflected M-Heun corrector became nonfinite")
    corr_lower_reflections=sum(Mrawnew .< zero(T))
    corr_upper_reflections=sum(Mrawnew .> Mmax)
    corr_raw_min=minimum(Mrawnew); corr_raw_max=maximum(Mrawnew)
    accepted_mean_M=sum(M)/T(n); accepted_max_M=maximum(M); accepted_min_M=minimum(M)
    println("[reflected-amplitude] step=",step_id,
        " M_old(min,mean,max)=",(old_min_M,old_mean_M,old_max_M),
        " M_new(min,mean,max)=",(accepted_min_M,accepted_mean_M,accepted_max_M),
        " reflections(predL,predU,corrL,corrU)=",
        (pred_lower_reflections,pred_upper_reflections,
         corr_lower_reflections,corr_upper_reflections))
    return (old_min_M=old_min_M,old_mean_M=old_mean_M,old_max_M=old_max_M,
        pred_mean_M=sum(Mp)/T(n),pred_max_M=maximum(Mp),
        accepted_min_M=accepted_min_M,accepted_mean_M=accepted_mean_M,
        accepted_max_M=accepted_max_M,
        predictor_lower_reflections=pred_lower_reflections,
        predictor_upper_reflections=pred_upper_reflections,
        predictor_raw_min=pred_raw_min,
        predictor_raw_max=pred_raw_max,
        corrector_lower_reflections=corr_lower_reflections,
        corrector_upper_reflections=corr_upper_reflections,
        corrector_raw_min=corr_raw_min,
        corrector_raw_max=corr_raw_max,
        endpoint_converged=endpoint_info.converged,
        endpoint_iterations=endpoint_info.iterations,
        endpoint_final_alpha=endpoint_info.final_alpha,
        endpoint_min_alpha=endpoint_info.min_alpha,
        endpoint_max_alpha=endpoint_info.max_alpha,
        endpoint_alpha_increases=endpoint_info.alpha_increases,
        endpoint_alpha_decreases=endpoint_info.alpha_decreases,
        midpoint_converged=midpoint_info.converged,
        midpoint_iterations=midpoint_info.iterations,
        midpoint_final_alpha=midpoint_info.final_alpha,
        midpoint_min_alpha=midpoint_info.min_alpha,
        midpoint_max_alpha=midpoint_info.max_alpha,
        midpoint_alpha_increases=midpoint_info.alpha_increases,
        midpoint_alpha_decreases=midpoint_info.alpha_decreases)
end

#writedlm(joinpath(base_path, "rand_m_k=$k.txt"), sqrt.(target_mx.^2+target_my.^2+target_mz.^2))
#writedlm(joinpath(base_path, "target_mx.txt"), target_mx)
#writedlm(joinpath(base_path, "target_my.txt"), target_my)
#writedlm(joinpath(base_path, "target_mz.txt"), target_mz)

# All target values are around zero and real, which implies hamiltonian is hermitian--> checked

# Initialize the local field
Bx,By,Bz = (2f0*CUDA.rand(REAL_T, nSt).-1f0)*k, (2f0*CUDA.rand(REAL_T, nSt).-1f0)*k, (2f0*CUDA.rand(REAL_T, nSt).-1f0)*k
lnn_x = CUDA.zeros(REAL_T, nSt)
lnn_y = CUDA.zeros(REAL_T, nSt)
lnn_z = CUDA.zeros(REAL_T, nSt)
temp_mx       = CUDA.zeros(REAL_T, nSt)
temp_my       = CUDA.zeros(REAL_T, nSt)
temp_mz       = CUDA.zeros(REAL_T, nSt)
hmn   = CUDA.zeros(COMPLEX_T, (2nSt, 2nSt))
fermiF   = CUDA.zeros(REAL_T, 2*nSt)

# Old-state amplitude and orientation.
M = similar(target_mx)
ex = similar(target_mx); ey = similar(target_my); ez = similar(target_mz)

# Reflected M-Heun predictor and corrector raw proposals.
Mp = similar(target_mx)
Mrawp = similar(target_mx)
Mrawnew = similar(target_mx)
epx = similar(target_mx); epy = similar(target_my); epz = similar(target_mz)
mxp = similar(target_mx); myp = similar(target_my); mzp = similar(target_mz)
lnn_xp = similar(lnn_x); lnn_yp = similar(lnn_y); lnn_zp = similar(lnn_z)
Bxp = similar(Bx); Byp = similar(By); Bzp = similar(Bz)

# Arithmetic old/predictor midpoint required by the second SIB stage.
mxs = similar(target_mx); mys = similar(target_my); mzs = similar(target_mz)
lnn_xs = similar(lnn_x); lnn_ys = similar(lnn_y); lnn_zs = similar(lnn_z)
Bxs = similar(Bx); Bys = similar(By); Bzs = similar(Bz)

# Gaussian buffers reused in both stages of each stochastic Heun step.
ξpar = similar(target_mx)
ξx = similar(target_mx); ξy = similar(target_my); ξz = similar(target_mz)

# Scratch magnetization for the predictor constrained solve.
temp_mx_cp = similar(temp_mx)
temp_my_cp = similar(temp_my)
temp_mz_cp = similar(temp_mz)

# Reused GPU workspace. Full histories are copied to disk only when a field
# solve reaches FIELD_MAX_ITERATIONS without satisfying the tolerance.
field_history = (
    alpha_site=CUDA.zeros(REAL_T,nSt),
    old_site_residual=CUDA.zeros(REAL_T,nSt),
    improvement_streak=CUDA.zeros(Int32,nSt),
    worsening_streak=CUDA.zeros(Int32,nSt),
    alpha_increases=CUDA.zeros(Int32,nSt),
    alpha_decreases=CUDA.zeros(Int32,nSt),
    delta_x=CUDA.zeros(REAL_T,nSt),
    delta_y=CUDA.zeros(REAL_T,nSt),
    delta_z=CUDA.zeros(REAL_T,nSt),
    hist_mx=CUDA.zeros(REAL_T,nSt,FIELD_MAX_ITERATIONS),
    hist_my=CUDA.zeros(REAL_T,nSt,FIELD_MAX_ITERATIONS),
    hist_mz=CUDA.zeros(REAL_T,nSt,FIELD_MAX_ITERATIONS),
    hist_lx=CUDA.zeros(REAL_T,nSt,FIELD_MAX_ITERATIONS),
    hist_ly=CUDA.zeros(REAL_T,nSt,FIELD_MAX_ITERATIONS),
    hist_lz=CUDA.zeros(REAL_T,nSt,FIELD_MAX_ITERATIONS),
    hist_alpha=CUDA.zeros(REAL_T,nSt,FIELD_MAX_ITERATIONS),
    hist_alpha_candidate=CUDA.fill(REAL_T(NaN),nSt,FIELD_MAX_ITERATIONS),
    hist_alignment=CUDA.fill(REAL_T(NaN),nSt,FIELD_MAX_ITERATIONS))

find_local_field!(dim, hmn, fermiF, nSt, fil, kT, tnn, stp,
    target_mx, target_my, target_mz, μ, k, Bx, By, Bz,
    lnn_x, lnn_y, lnn_z, temp_mx, temp_my, temp_mz;
    stage="initial accepted state",history=field_history)
nRuns = parse(Int, get(ENV, "SDW_NRUNS", "10000"))
nRuns > 0 || error("SDW_NRUNS must be a positive integer")
diagnostic_path = joinpath(experimental_path, "M_reflected_field_diagnostics.csv")
open(diagnostic_path, "w") do io
    println(io, "step,old_min_M,old_mean_M,old_max_M,pred_mean_M,pred_max_M," *
        "accepted_min_M,accepted_mean_M,accepted_max_M," *
        "predictor_lower_reflections,predictor_upper_reflections," *
        "predictor_raw_min,predictor_raw_max," *
        "corrector_lower_reflections,corrector_upper_reflections," *
        "corrector_raw_min,corrector_raw_max," *
        "endpoint_converged,endpoint_field_iterations,endpoint_final_alpha,endpoint_min_alpha,endpoint_max_alpha," *
        "endpoint_alpha_increases,endpoint_alpha_decreases," *
        "midpoint_converged,midpoint_field_iterations,midpoint_final_alpha,midpoint_min_alpha,midpoint_max_alpha," *
        "midpoint_alpha_increases,midpoint_alpha_decreases," *
        "accepted_converged,accepted_field_iterations,accepted_final_alpha,accepted_min_alpha,accepted_max_alpha," *
        "accepted_alpha_increases,accepted_alpha_decreases," *
        "accepted_residual_x,accepted_residual_y," *
        "accepted_residual_z")
end
#mx_all_gpu = CUDA.zeros(Float64, nSt, nRuns)
#my_all_gpu = CUDA.zeros(Float64, nSt, nRuns)
#mz_all_gpu = CUDA.zeros(Float64, nSt, nRuns)

for run = 1:nRuns
    mdiag = glsd_mheun_sib_reflected_step!(target_mx,target_my,target_mz,
        dt,Γpar,Γperp,kT,kT_fluc,lnn_x,lnn_y,lnn_z,Bx,By,Bz,
        M,ex,ey,ez,Mp,Mrawp,epx,epy,epz,mxp,myp,mzp,
        lnn_xp,lnn_yp,lnn_zp,Bxp,Byp,Bzp,
        mxs,mys,mzs,lnn_xs,lnn_ys,lnn_zs,Bxs,Bys,Bzs,
        ξpar,ξx,ξy,ξz,Mrawnew,
        hmn,fermiF,temp_mx_cp,temp_my_cp,temp_mz_cp,field_history,run)

    #global lnn_x, lnn_y, lnn_z, vals, vecs, Bx, By, Bz, run_iter,Energy =
    #    find_local_field(dim, hmn, fermiF, nSt, fil, kT, tnn, stp, target_mx, target_my, target_mz, μ, base_path,k,Bx,By,Bz,lnn_x,lnn_y,lnn_z,temp_mx,temp_my,temp_mz)
    accepted_info = find_local_field!(dim, hmn, fermiF, nSt, fil, kT, tnn, stp,
        target_mx, target_my, target_mz, μ, k, Bx, By, Bz,
        lnn_x, lnn_y, lnn_z, temp_mx, temp_my, temp_mz;
        stage="accepted state, step $run",history=field_history)

    open(diagnostic_path,"a") do io
        println(io,join((run,mdiag.old_min_M,mdiag.old_mean_M,mdiag.old_max_M,
            mdiag.pred_mean_M,mdiag.pred_max_M,mdiag.accepted_min_M,
            mdiag.accepted_mean_M,mdiag.accepted_max_M,
            mdiag.predictor_lower_reflections,mdiag.predictor_upper_reflections,
            mdiag.predictor_raw_min,mdiag.predictor_raw_max,
            mdiag.corrector_lower_reflections,mdiag.corrector_upper_reflections,
            mdiag.corrector_raw_min,mdiag.corrector_raw_max,
            mdiag.endpoint_converged,mdiag.endpoint_iterations,mdiag.endpoint_final_alpha,
            mdiag.endpoint_min_alpha,mdiag.endpoint_max_alpha,
            mdiag.endpoint_alpha_increases,mdiag.endpoint_alpha_decreases,
            mdiag.midpoint_converged,mdiag.midpoint_iterations,mdiag.midpoint_final_alpha,
            mdiag.midpoint_min_alpha,mdiag.midpoint_max_alpha,
            mdiag.midpoint_alpha_increases,mdiag.midpoint_alpha_decreases,
            accepted_info.converged,accepted_info.iterations,accepted_info.final_alpha,
            accepted_info.min_alpha,accepted_info.max_alpha,
            accepted_info.alpha_increases,accepted_info.alpha_decreases,
            accepted_info.residual_x,
            accepted_info.residual_y,accepted_info.residual_z),',')); flush(io)
    end

    #@inbounds CUDA.@sync mx_all_gpu[:, run] .= target_mx
    #@inbounds CUDA.@sync my_all_gpu[:, run] .= target_my
    #@inbounds CUDA.@sync mz_all_gpu[:, run] .= target_mz
    if run % 10 == 0
        # Save the actual evolved collective variables.  temp_m* are scratch
        # electronic expectation values from the constrained field solve.
        writedlm(joinpath(experimental_path, "temp_mx_$run.txt"), Array(target_mx))
        writedlm(joinpath(experimental_path, "temp_my_$run.txt"), Array(target_my))
        writedlm(joinpath(experimental_path, "temp_mz_$run.txt"), Array(target_mz))
    end
    #writedlm(joinpath(base_path, "Bx_$run.txt"), Bx)
    #writedlm(joinpath(base_path, "By_$run.txt"), By)
    #writedlm(joinpath(base_path, "Bz_$run.txt"), Bz)
    #writedlm(joinpath(base_path, "lnnx_$run.txt"), lnn_x)
    #writedlm(joinpath(base_path, "lnny_$run.txt"), lnn_y)
    #writedlm(joinpath(base_path, "lnnz_$run.txt"), lnn_z)
    #writedlm(joinpath(base_path, "run_$run.txt"), run_iter)
    #writedlm(joinpath(base_path, "Energy_$run.txt"), Energy)
    
end
open(joinpath(experimental_path, "run_status.txt"), "a") do io
    println(io, "status=completed")
    println(io, "completed_at=$(Dates.format(now(), dateformat\"yyyy-mm-ddTHH:MM:SS\"))")
    println(io, "completed_dynamics_steps=$nRuns")
end
#mx_all = Array(mx_all_gpu)
#my_all = Array(my_all_gpu)
#mz_all = Array(mz_all_gpu)
