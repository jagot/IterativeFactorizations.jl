module MatrixFactorizations

using LinearAlgebra

using IterativeSolvers
import IterativeSolvers: GMRESIterable, gmres_iterable!
using AlgebraicMultigrid
using SparseArrays

const ConjugateGradient = Union{CGIterable,PCGIterable}

LinearAlgebra.isposdef(T::Union{Tridiagonal,SymTridiagonal}) = isposdef(Matrix(T))

mutable struct IterativeFactorization{Iterator,X,B,T}
    iterator::Iterator
    x::X # Solution vector
    b::B # Right-hand side vector
    tol::T
end

Base.size(A::IterativeFactorization) = (length(A.b), length(A.x))
Base.size(A::IterativeFactorization, i::Int) = size(A)[i]
Base.eltype(A::IterativeFactorization) = eltype(A.x)

getscalarresidual(it::ConjugateGradient) = it.residual
getscalarresidual(it::GMRESIterable) = it.residual.current

preconditioner(Pl::P) where {P<:AbstractMatrix} =
    aspreconditioner(ruge_stuben(sparse(Pl)))
preconditioner(Pl::SymTridiagonal) = factorize(Pl)

function IterativeFactorization(A::M,
                                x::X=zeros(eltype(A), size(A, 2)),
                                b::B=similar(x);
                                tol=√(eps(real(eltype(b)))),
                                kwargs...) where {M<:AbstractMatrix,X,B}
    prec = preconditioner(A)
    # prec = aspreconditioner(ruge_stuben(sparse(A)))
    # prec = Identity()

    iterator,b = if isposdef(A)
        iterator = cg_iterator!(x, A, b, prec; tol=tol, initially_zero=iszero(x), kwargs...)
        iterator,iterator.r
    else
        iterator = gmres_iterable!(x, A, b; Pl=prec, tol=tol,
                                   initially_zero=iszero(x), kwargs...)
        iterator,iterator.b
    end

    IterativeFactorization(iterator, x, b, tol)
end

update_preconditioner!(A::IterativeFactorization, Pl::P) where {P<:AbstractMatrix} =
    A.iterator.Pl = preconditioner(Pl)

function reset_iterator!(it::It, tol::T) where {It<:ConjugateGradient,T}
    # It is assumed that the new RHS is already copied to it.r

    initially_zero = iszero(it.x)
    it.mv_products = if initially_zero
        0
    else
        mul!(it.c, it.A, it.x)
        it.r .-= it.c
        1
    end

    it.residual = norm(it.r)
    it.reltol = tol*it.residual
    it.u .= zero(eltype(it.x))
end

function reset_iterator!(it::It, tol::T) where {It<:GMRESIterable, T}
    # It is assumed that the new RHS is already copied to it.b
    initially_zero=iszero(it.x)
    it.mv_products = initially_zero ? 0 : 1

    β = IterativeSolvers.init!(it.arnoldi, it.x, it.b, it.Pl, it.Ax,
                               initially_zero=initially_zero)
    it.residual.current = β
    IterativeSolvers.init_residual!(it.residual, β)
    it.reltol = tol * β
    it.β = β
end

function LinearAlgebra.ldiv!(x, A::IterativeFactorization, b; verbosity=0)
    iterator = A.iterator
    copyto!(A.x, x)
    copyto!(A.b, b)
    reset_iterator!(iterator, A.tol)

    verbosity > 0 && println("Initial residual: $(getscalarresidual(iterator))")

    ii = 0
    for (iteration,item) in enumerate(iterator)
        iterator isa ConjugateGradient && (iterator.mv_products += 1)
        verbosity > 1 && println("#$iteration: $(getscalarresidual(iterator))")
        ii += 1
    end
    verbosity > 0 && println("Solution converged: ",
                            IterativeSolvers.converged(iterator) ? "yes" : "no",
                            ", #iterations: ", ii, "/", iterator.maxiter,
                            ", residual: ", getscalarresidual(iterator))

    copyto!(x, iterator.x)
    x
end

export IterativeFactorization

end # module
