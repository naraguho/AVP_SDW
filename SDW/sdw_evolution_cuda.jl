using CUDA
using CUDA.CUSOLVER
#using CUDA.CUSOLVER: heevd
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics
# Cuda.jl only supports real symmetric matrix to get eigenvalue

const REAL_T = Float32
const COMPLEX_T = ComplexF32
const CONVERGENCE_TOL = 1f-5


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


function find_local_field!(dim, hmn, fermiF, nSt, fil, kT, tnn,  stp, target_mx, target_my, target_mz, μ,k,Bx,By,Bz,lnn_x,lnn_y,lnn_z,temp_mx,temp_my,temp_mz)
    ## initialize the system with random onsite potential
    #Bx,By,Bz = 2*rand(Float64, nSt).-1, 2*rand(Float64, nSt).-1, 2*rand(Float64, nSt).-1
    #Bx,By,Bz = (2*rand(Float64, nSt).-1)*k, (2*rand(Float64, nSt).-1)*k, (2*rand(Float64, nSt).-1)*k

    @cuda threads = nT blocks = nB init_sdw_hmn_kernel!(nSt, spin_dof, tnn, hmn, nnLu, nnRu, nnTu, nnBu, Bx,By,Bz)
    #hmn_cpu = Matrix(hmn)
    hmn_copy = copy(hmn)
    valsInit, vecsInit = CUDA.CUSOLVER.heevd!('V', 'U', hmn_copy)
    #valsInit = CuArray(valsInit)
    #vecsInit = CuArray(vecsInit)

    μ = find_mu_cuda!(nSt, valsInit, kT, REAL_T(2*nSt)*fil, fermiF)
    @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, valsInit, fermiF)
    @cuda threads = nT blocks = nB comp_m_kernel!(nSt, vecsInit, fermiF, temp_mx, temp_my, temp_mz)
    #GC.gc()
    #CUDA.reclaim()
    # mx is new one, target_mx is old one
    #println(temp_mx-target_mx)
    #println(temp_my-target_my)
    #println(temp_mz-target_mz)
    println(isa(temp_mx, CuArray))
    println(isa(target_mx, CuArray))
    println(CUDA.device(temp_mx) == CUDA.device(target_mx))
    diff = temp_mx .- target_mx
    val = sqrt(sum(abs2, diff))
    #println("Manual norm: ", val)
    #println("First norm is started")
    oldDiff_x = sqrt(sum(abs2, temp_mx .- target_mx))
    oldDiff_y = sqrt(sum(abs2, temp_my .- target_my))
    oldDiff_z = sqrt(sum(abs2, temp_mz .- target_mz))
    #println("First norm is done")

    delta_x  = stp * (temp_mx - target_mx)
    delta_y  = stp * (temp_my - target_my)
    delta_z  = stp * (temp_mz - target_mz)
    #global run_iter = 0
    #println(oldDiff,",",norm(delta))
    converged = false
    for run in 1:5000
        #GC.gc()
        #CUDA.reclaim()
        Bx .-= delta_x
        By .-= delta_y
        Bz .-= delta_z

        @cuda threads = nT blocks = nB updt_sdw_hmn_kernel!(nSt, spin_dof, tnn, hmn, nnLu, nnRu, nnTu, nnBu, Bx, By, Bz)
        #hmn_cpu = Matrix(hmn)
        hmn_copy = copy(hmn)
        vals, vecs = CUDA.CUSOLVER.heevd!('V', 'U', hmn_copy)
        #vals = CuArray(vals)
        #vecs = CuArray(vecs)
        μ = find_mu_cuda!(nSt, vals, kT, REAL_T(2*nSt)*fil, fermiF)
        @cuda threads = nT blocks = nB comp_fermi_kernel!(nSt, μ, kT, vals, fermiF)
        @cuda threads = nT blocks = nB comp_m_kernel!(nSt, vecs, fermiF, temp_mx, temp_my, temp_mz)

        newDiff_x = sqrt(sum(abs2, temp_mx .- target_mx))
        newDiff_y = sqrt(sum(abs2, temp_my .- target_my))
        newDiff_z = sqrt(sum(abs2, temp_mz .- target_mz))
        if (newDiff_x < CONVERGENCE_TOL) && (newDiff_y < CONVERGENCE_TOL) && (newDiff_z < CONVERGENCE_TOL)
            println("converged at ",run," run with precision in x ", newDiff_x, " precision in y ", newDiff_y," precision in z ",newDiff_z)
            println("Success!!")
            converged = true
            #run_iter = run
            break
        end
        oldDiff_x = newDiff_x
        oldDiff_y = newDiff_y
        oldDiff_z = newDiff_z
        delta_x  = stp * (temp_mx - target_mx)
        delta_y  = stp * (temp_my - target_my)
        delta_z  = stp * (temp_mz - target_mz)

        #if run%10 == 0
        #    println("x difference ", temp_mx[1]-target_mx[1])
        #end

    end
    converged || error("Constrained field solve failed to converge in 5000 iterations")

    #vals, vecs = eigen(hmn)
    #comp_fermi!(nSt, μ, kT, vals, fermiF)
    #comp_m!(nSt, vecs, fermiF, mx, my, mz)
    @cuda threads = nT blocks = nB comp_Lagrange_kernel!(nSt, Bx,By,Bz, nnLu, nnRu, nnBu, nnTu, lnn_x, lnn_y,lnn_z,temp_mx,temp_my,temp_mz)
    #Energy = comp_energy(nSt, vecs, fermiF, tnn, spin_dof, Bx, By, Bz, nnRu, nnTu)
    return nothing
end

dim   = 30
nSt   = dim * dim
# Initial value : half-filling
fil   = 0.5f0
kT    = 0.05f0   # electronic Fermi-Dirac smearing temperature
kT_fluc = 0.0005f0 # stochastic fluctuation/bath temperature used only in Heun/SIB noise
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
println("Hi")
spin_dof = 2
# neighborhood site index
nnLu,nnRu,nnBu,nnTu = computeNN_cuda(nT, nB, dim, nSt, spin_dof)
println("Before SC loop")
μ,Δ = selfConsistent(dim, nSt, spin_dof, kT, fil, tnn,Δ)
println("Converged chemical potential: ", μ, " converged order parameter: ", Δ)
println("Electronic smearing kT = ", kT, "; fluctuation temperature kT_fluc = ", kT_fluc)

base_path = get(ENV, "SDW_OUTPUT_DIR",
    "/home/sgv2ew/AdiabaticSDW/Neel30-new_Telec0.05_Tfluc0.0005_Gpar0.1/")
mkpath(base_path)

target_mx, target_my, target_mz = initialize_random_magnetization(nSt, tnn, kT, μ, spin_dof,k)

#init_Neel!(nSt, dim, fil, 0.3, target_mx, target_my, target_mz)
#println("target_mx", target_mx)
#println("target_my", target_my)
#println("target_mz", target_mz)

# The bounded amplitude coordinate x, M=(1/2)tanh(x), is advanced with
# multiplicative-noise stochastic Heun in the Stratonovich convention.  The
# unit direction e is advanced with the two length-preserving stages of SIB.

@inline function cayley_rotate(ex, ey, ez, qx, qy, qz)
    T = typeof(ex)
    half = T(0.5)
    two = T(2)
    wx, wy, wz = half*qx, half*qy, half*qz
    w2 = wx*wx + wy*wy + wz*wz
    edotw = ex*wx + ey*wy + ez*wz
    cx = ey*wz - ez*wy
    cy = ez*wx - ex*wz
    cz = ex*wy - ey*wx
    den = one(T) + w2
    common = one(T) - w2
    nx = (common*ex + two*cx + two*edotw*wx) / den
    ny = (common*ey + two*cy + two*edotw*wy) / den
    nz = (common*ez + two*cz + two*edotw*wz) / den
    return nx, ny, nz
end

function decompose_m_kernel!(nSt, mx, my, mz, M, xamp, ex, ey, ez, epsm, xmax)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        T = eltype(mx)
        x, y, z = mx[idx], my[idx], mz[idx]
        mag = sqrt(x*x + y*y + z*z)
        M[idx] = mag
        # Exact interior inverse of M=(1/2)tanh(x).  A NaN deliberately aborts
        # the step if the supplied state is outside the physical interval.
        arg = T(2)*mag
        xamp[idx] = arg < one(T) ? atanh(arg) : T(NaN)
        xamp[idx] = xamp[idx] <= xmax ? xamp[idx] : T(NaN)
        if mag > epsm
            invmag = one(T)/mag
            ex[idx], ey[idx], ez[idx] = x*invmag, y*invmag, z*invmag
        else
            # Orientation is undefined at M=0.  Keep a deterministic fallback;
            # the branch is counted after the run by monitoring min(|m|).
            ex[idx], ey[idx], ez[idx] = zero(T), zero(T), one(T)
        end
    end
    return nothing
end

function xheun_sib_predictor_kernel!(nSt, dt, Γpar, Γperp, kT_fluc,
    epsm, xmax, M, xamp, ex, ey, ez, bx, by, bz,
    ξpar, ξx, ξy, ξz, Mp, xpamp, epx, epy, epz, mxp, myp, mzp)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        T = eltype(M)
        two = T(2)
        eight = T(8)
        mag = M[idx]
        xold = xamp[idx]
        ux, uy, uz = ex[idx], ey[idx], ez[idx]
        fx, fy, fz = bx[idx], by[idx], bz[idx]

        # Exact Stratonovich x equation for M=(1/2)tanh(x):
        # dx = 2 Γpar cosh(x)^2 b_parallel dt
        #    + sqrt(8 kT Γpar) cosh(x)^2 o dW.
        fpar = ux*fx + uy*fy + uz*fz
        c2old = cosh(xold)^2
        drift_x0 = two*Γpar*c2old*fpar
        diff_x0 = sqrt(max(eight*kT_fluc*Γpar, zero(T)))*c2old
        xpred = xold + dt*drift_x0 + sqrt(dt)*diff_x0*ξpar[idx]

        # M is a magnitude, so M=0 is a reflecting boundary.  Reflection at
        # x=0 is separate from (and does not alter) the exact interior change
        # of variables derived in the accompanying note.
        xpred = abs(xpred)
        xpred = xpred <= xmax ? xpred : T(NaN)
        magp = T(0.5)*tanh(xpred)

        # First SIB stage:
        # ep-e = ((e+ep)/2) x q(e,M,b,DeltaW).
        if mag > epsm
            ebx = uy*fz - uz*fy
            eby = uz*fx - ux*fz
            ebz = ux*fy - uy*fx
            ewx = uy*ξz[idx] - uz*ξy[idx]
            ewy = uz*ξx[idx] - ux*ξz[idx]
            ewz = ux*ξy[idx] - uy*ξx[idx]
            gamma = Γperp/mag
            sigma_perp = sqrt(max(two*kT_fluc*Γperp*dt, zero(T)))/mag
            qx = dt*(fx - gamma*ebx) - sigma_perp*ewx
            qy = dt*(fy - gamma*eby) - sigma_perp*ewy
            qz = dt*(fz - gamma*ebz) - sigma_perp*ewz
            px, py, pz = cayley_rotate(ux, uy, uz, qx, qy, qz)
        else
            px, py, pz = ux, uy, uz
        end

        Mp[idx] = magp
        xpamp[idx] = xpred
        epx[idx], epy[idx], epz[idx] = px, py, pz
        mxp[idx], myp[idx], mzp[idx] = magp*px, magp*py, magp*pz
    end
    return nothing
end

function midpoint_m_kernel!(nSt, M, ex, ey, ez, Mp, epx, epy, epz,
                            mxs, mys, mzs)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        T = eltype(M)
        half = T(0.5)
        magh = half*(M[idx] + Mp[idx])
        ehx = half*(ex[idx] + epx[idx])
        ehy = half*(ey[idx] + epy[idx])
        ehz = half*(ez[idx] + epz[idx])
        mxs[idx] = magh*ehx
        mys[idx] = magh*ehy
        mzs[idx] = magh*ehz
    end
    return nothing
end

function xheun_sib_corrector_kernel!(nSt, dt, Γpar, Γperp, kT_fluc,
    epsm, xmax, M, xamp, ex, ey, ez, bx, by, bz,
    Mp, xpamp, epx, epy, epz, bxp, byp, bzp, bxs, bys, bzs,
    ξpar, ξx, ξy, ξz, mx, my, mz)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nSt
        T = eltype(M)
        half = T(0.5)
        two = T(2)
        eight = T(8)
        mag = M[idx]
        magp = Mp[idx]
        xold = xamp[idx]
        xpred = xpamp[idx]
        ux, uy, uz = ex[idx], ey[idx], ez[idx]
        px, py, pz = epx[idx], epy[idx], epz[idx]

        # Stochastic Heun for multiplicative Stratonovich noise in x.  The
        # endpoint drift and diffusion are both reevaluated at the joint
        # predictor (xpred, ep, bpred), using the same Wiener increment.
        c2old = cosh(xold)^2
        c2pred = cosh(xpred)^2
        fpar0 = ux*bx[idx] + uy*by[idx] + uz*bz[idx]
        fparp = px*bxp[idx] + py*byp[idx] + pz*bzp[idx]
        drift_x0 = two*Γpar*c2old*fpar0
        drift_xp = two*Γpar*c2pred*fparp
        sigma0 = sqrt(max(eight*kT_fluc*Γpar, zero(T)))*c2old
        sigmap = sqrt(max(eight*kT_fluc*Γpar, zero(T)))*c2pred
        xnew = xold + half*dt*(drift_x0 + drift_xp) +
               half*sqrt(dt)*(sigma0 + sigmap)*ξpar[idx]
        xnew = abs(xnew)
        xnew = xnew <= xmax ? xnew : T(NaN)
        magnew = T(0.5)*tanh(xnew)

        # Second SIB stage.  Coefficients are evaluated at the arithmetic
        # old/predictor midpoint, as the split-variable analogue of Eq. (18), while the implicit
        # midpoint containing e_new is solved analytically by the Cayley map.
        ehx = half*(ux + px)
        ehy = half*(uy + py)
        ehz = half*(uz + pz)
        magh = half*(mag + magp)
        fx, fy, fz = bxs[idx], bys[idx], bzs[idx]

        if magh > epsm
            ebx = ehy*fz - ehz*fy
            eby = ehz*fx - ehx*fz
            ebz = ehx*fy - ehy*fx
            ewx = ehy*ξz[idx] - ehz*ξy[idx]
            ewy = ehz*ξx[idx] - ehx*ξz[idx]
            ewz = ehx*ξy[idx] - ehy*ξx[idx]
            gamma = Γperp/magh
            sigma_perp = sqrt(max(two*kT_fluc*Γperp*dt, zero(T)))/magh
            qx = dt*(fx - gamma*ebx) - sigma_perp*ewx
            qy = dt*(fy - gamma*eby) - sigma_perp*ewy
            qz = dt*(fz - gamma*ebz) - sigma_perp*ewz
            nx, ny, nz = cayley_rotate(ux, uy, uz, qx, qy, qz)
        else
            nx, ny, nz = ux, uy, uz
        end

        mx[idx], my[idx], mz[idx] = magnew*nx, magnew*ny, magnew*nz
    end
    return nothing
end

function glsd_xheun_sib_step!(mx, my, mz, dt, Γpar, Γperp, kT_elec, kT_fluc,
    bx, by, bz, Bfield_x, Bfield_y, Bfield_z,
    M, xamp, ex, ey, ez, Mp, xpamp, epx, epy, epz,
    mxp, myp, mzp, bxp, byp, bzp, Bxp, Byp, Bzp,
    mxs, mys, mzs, bxs, bys, bzs, Bxs, Bys, Bzs,
    ξpar, ξx, ξy, ξz,
    hmn, fermiF, temp_mx_cp, temp_my_cp, temp_mz_cp)

    T = eltype(mx)
    n = length(mx)
    epsm = T(1e-7)
    # Beyond this point the transformed mobility 4Γ cosh(x)^4 is so large
    # that Float32 integration is not trustworthy.  Abort instead of silently
    # clipping x or M; reduce dt if this guard is reached.
    xmax = T(8)
    dt > zero(T) || error("dt must be positive")
    kT_fluc >= zero(T) || error("The fluctuation temperature must be nonnegative")
    Γpar >= zero(T) || error("Gamma_parallel must be nonnegative")
    Γperp >= zero(T) || error("Gamma_perp must be nonnegative")

    ξpar .= CUDA.randn(T, n)
    ξx   .= CUDA.randn(T, n)
    ξy   .= CUDA.randn(T, n)
    ξz   .= CUDA.randn(T, n)

    @cuda threads=nT blocks=nB decompose_m_kernel!(
        nSt, mx, my, mz, M, xamp, ex, ey, ez, epsm, xmax)
    any(.!isfinite.(xamp)) && error("Invalid amplitude: require 0 <= M < 1/2 and x <= $xmax")

    # Do not impose an arbitrary local-noise cutoff here.  The quantity
    # sqrt(8*T*Gamma_parallel*dt)*cosh(x)^2 is useful to *report* in a
    # timestep-convergence study, but a single large value is not a failed
    # electronic self-consistency solve and should not terminate production.
    # The hard protections below remain: invalid M>=1/2, x>xmax, and nonfinite
    # predictor/corrector values stop the run explicitly.

    # Heun amplitude predictor plus the first length-preserving SIB stage.
    @cuda threads=nT blocks=nB xheun_sib_predictor_kernel!(
        nSt, dt, Γpar, Γperp, kT_fluc, epsm, xmax,
        M, xamp, ex, ey, ez, bx, by, bz, ξpar, ξx, ξy, ξz,
        Mp, xpamp, epx, epy, epz, mxp, myp, mzp)
    any(.!isfinite.(xpamp)) && error("x-Heun predictor became stiff/nonfinite; reduce dt")

    # Predictor field for the amplitude-Heun corrector.
    copyto!(Bxp, Bfield_x); copyto!(Byp, Bfield_y); copyto!(Bzp, Bfield_z)
    find_local_field!(dim, hmn, fermiF, nSt, fil, kT_elec, tnn, stp,
        mxp, myp, mzp, μ, k, Bxp, Byp, Bzp,
        bxp, byp, bzp, temp_mx_cp, temp_my_cp, temp_mz_cp)

    # Eq. (18) evaluates the SIB corrector coefficients at
    # the separate amplitude and orientation midpoints.
    @cuda threads=nT blocks=nB midpoint_m_kernel!(
        nSt, M, ex, ey, ez, Mp, epx, epy, epz, mxs, mys, mzs)
    Bxs .= T(0.5) .* (Bfield_x .+ Bxp)
    Bys .= T(0.5) .* (Bfield_y .+ Byp)
    Bzs .= T(0.5) .* (Bfield_z .+ Bzp)
    find_local_field!(dim, hmn, fermiF, nSt, fil, kT_elec, tnn, stp,
        mxs, mys, mzs, μ, k, Bxs, Bys, Bzs,
        bxs, bys, bzs, temp_mx_cp, temp_my_cp, temp_mz_cp)

    # Heun amplitude corrector plus the second SIB stage.  The same four
    # Gaussian arrays are reused, as required for the Stratonovich step.
    @cuda threads=nT blocks=nB xheun_sib_corrector_kernel!(
        nSt, dt, Γpar, Γperp, kT_fluc, epsm, xmax,
        M, xamp, ex, ey, ez, bx, by, bz,
        Mp, xpamp, epx, epy, epz, bxp, byp, bzp, bxs, bys, bzs,
        ξpar, ξx, ξy, ξz, mx, my, mz)
    any(.!isfinite.(mx)) && error("x-Heun corrector became stiff/nonfinite; reduce dt")
    return nothing
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
xamp = similar(target_mx)
ex = similar(target_mx); ey = similar(target_my); ez = similar(target_mz)

# SIB/Heun predictor amplitude, orientation, moment, and field.
Mp = similar(target_mx)
xpamp = similar(target_mx)
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

find_local_field!(dim, hmn, fermiF, nSt, fil, kT, tnn, stp, target_mx, target_my, target_mz, μ, k,Bx,By,Bz,lnn_x,lnn_y,lnn_z,temp_mx,temp_my,temp_mz)
nRuns = parse(Int, get(ENV, "SDW_NRUNS", "10000"))
nRuns > 0 || error("SDW_NRUNS must be a positive integer")
#mx_all_gpu = CUDA.zeros(Float64, nSt, nRuns)
#my_all_gpu = CUDA.zeros(Float64, nSt, nRuns)
#mz_all_gpu = CUDA.zeros(Float64, nSt, nRuns)

for run = 1:nRuns
    glsd_xheun_sib_step!(target_mx, target_my, target_mz,
        dt, Γpar, Γperp, kT, kT_fluc,
        lnn_x, lnn_y, lnn_z, Bx, By, Bz,
        M, xamp, ex, ey, ez, Mp, xpamp, epx, epy, epz,
        mxp, myp, mzp, lnn_xp, lnn_yp, lnn_zp, Bxp, Byp, Bzp,
        mxs, mys, mzs, lnn_xs, lnn_ys, lnn_zs, Bxs, Bys, Bzs,
        ξpar, ξx, ξy, ξz,
        hmn, fermiF, temp_mx_cp, temp_my_cp, temp_mz_cp)

    #global lnn_x, lnn_y, lnn_z, vals, vecs, Bx, By, Bz, run_iter,Energy =
    #    find_local_field(dim, hmn, fermiF, nSt, fil, kT, tnn, stp, target_mx, target_my, target_mz, μ, base_path,k,Bx,By,Bz,lnn_x,lnn_y,lnn_z,temp_mx,temp_my,temp_mz)
    find_local_field!(dim, hmn, fermiF, nSt, fil, kT, tnn, stp, target_mx, target_my, target_mz, μ, k,Bx,By,Bz,lnn_x,lnn_y,lnn_z,temp_mx,temp_my,temp_mz)

    #@inbounds CUDA.@sync mx_all_gpu[:, run] .= target_mx
    #@inbounds CUDA.@sync my_all_gpu[:, run] .= target_my
    #@inbounds CUDA.@sync mz_all_gpu[:, run] .= target_mz
    if run % 10 == 0
        # Save the actual evolved collective variables.  temp_m* are scratch
        # electronic expectation values from the constrained field solve.
        writedlm(joinpath(base_path, "temp_mx_$run.txt"), Array(target_mx))
        writedlm(joinpath(base_path, "temp_my_$run.txt"), Array(target_my))
        writedlm(joinpath(base_path, "temp_mz_$run.txt"), Array(target_mz))
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
#mx_all = Array(mx_all_gpu)
#my_all = Array(my_all_gpu)
#mz_all = Array(mz_all_gpu)
