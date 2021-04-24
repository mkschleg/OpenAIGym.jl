module OpenAIGym

using PyCall
import PyCall: hasproperty
using Reexport
@reexport using MinimalRLCore
import MinimalRLCore: AbstractEnvironment
# @reexport using Reinforce
# import Reinforce:
#     MouseAction, MouseActionSet,
#     KeyboardAction, KeyboardActionSet

export
    GymEnv,
    render,
    close


struct DiscreteSet{T<:AbstractArray} <: AbstractSet{T}
    items::T
end
randtype(s::DiscreteSet) = eltype(s.items)
Base.rand(s::DiscreteSet, dims::Integer...) = rand(s.items, dims...)
Base.in(x, s::DiscreteSet) = x in s.items
Base.length(s::DiscreteSet) = length(s.items)
Base.getindex(s::DiscreteSet, i::Int) = s.items[i]
Base.:(==)(s1::DiscreteSet, s2::DiscreteSet) = s1.items == s2.items
Base.isempty(s::DiscreteSet) = Base.isempty(s.items)

# --------------------------------------------------------------

abstract type AbstractGymEnv <: AbstractEnvironment end

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
mutable struct GymEnv{T} <: AbstractGymEnv
    name::Symbol
    ver::Symbol
    pyenv::PyObject   # the python "env" object
    pystep::PyObject  # the python env.step function
    pyreset::PyObject # the python env.reset function
    pystate::PyObject # the state array object referenced by the PyArray state.o
    pystepres::PyObject # used to make stepping the env slightly more efficient
    info::PyObject    # store it as a PyObject for speed, since often unused
    state::T
    reward::Float64
    total_reward::Float64
    actions::AbstractSet
    done::Bool
    function GymEnv{T}(name, ver, pyenv, pystate, state, seed) where T
        env = new{T}(name, ver, pyenv, pyenv."step", pyenv."reset",
                     pystate, PyNULL(), PyNULL(), state)
        env.pyenv.seed(seed)
        MinimalRLCore.reset!(env)
        env
    end
end

use_pyarray_state(envname::Symbol) = !(envname ∈ (:Blackjack,))

function GymEnv(name::Symbol, ver::Symbol = :v0;
                stateT = ifelse(use_pyarray_state(name), PyArray, PyAny),
                seed=0)
    if PyCall.ispynull(pysoccer) && name ∈ (:Soccer, :SoccerEmptyGoal)
        copy!(pysoccer, pyimport("gym_soccer"))
    end

    GymEnv(name, ver, pygym.make("$name-$ver"), stateT, seed)
end

GymEnv(name::AbstractString; kwargs...) =
    GymEnv(Symbol.(split(name, '-', limit = 2))...; kwargs...)

function GymEnv(name::Symbol, ver::Symbol, pyenv, stateT, seed)
    pystate = pycall(pyenv."reset", PyObject)
    state = convert(stateT, pystate)
    T = typeof(state)
    GymEnv{T}(name, ver, pyenv, pystate, state, seed)
end

function Base.show(io::IO, env::GymEnv)
  println(io, "GymEnv $(env.name)-$(env.ver)")
  if hasproperty(env.pyenv, :class_name)
    println(io, "  $(env.pyenv.class_name())")
  end
  println(io, "  r  = $(env.reward)")
  print(  io, "  ∑r = $(env.total_reward)")
end

# --------------------------------------------------------------

"""
    close(env::AbstractGymEnv)
"""
Base.close(env::AbstractGymEnv) =
	!ispynull(env.pyenv) && env.pyenv.close()

# --------------------------------------------------------------

"""
    render(env::AbstractGymEnv; mode = :human)

# Arguments

- `mode`: `:human`, `:rgb_array`, `:ansi`
"""
render(env::AbstractGymEnv, args...; kwargs...) =
    pycall(env.pyenv.render, PyAny; kwargs...)

# --------------------------------------------------------------


function actionset(A::PyObject)
    if hasproperty(A, :n)
        # choose from n actions
        DiscreteSet(0:A.n-1)
    elseif hasproperty(A, :spaces)
        # a tuple of action sets
        sets = [actionset(a) for a in A.spaces]
        TupleSet(sets...)
    elseif hasproperty(A, :high)
        # continuous interval
        IntervalSet{Vector{Float64}}(A.low, A.high)
        # if A[:shape] == (1,)  # for now we only support 1-length vectors
        #     IntervalSet{Float64}(A[:low][1], A[:high][1])
        # else
        #     # @show A[:shape]
        #     lo,hi = A[:low], A[:high]
        #     # error("Unsupported shape for IntervalSet: $(A[:shape])")
        #     [IntervalSet{Float64}(lo[i], hi[i]) for i=1:length(lo)]
        # end
    elseif hasproperty(A, :actions)
        # Hardcoded
        TupleSet(DiscreteSet(A.actions))
    else
        @show A
        @show keys(A)
        error("Unknown actionset type: $A")
    end
end

function MinimalRLCore.get_actions(env::AbstractGymEnv, s′)
    actionset(env.pyenv.action_space)
end

pyaction(a::Vector) = Any[pyaction(ai) for ai=a]
pyaction(a) = a

"""
`reset!(env::GymEnv)` reset the environment
"""
function MinimalRLCore.reset!(env::GymEnv, args...)
    pycall!(env.pystate, env.pyreset, PyObject)
    convert_state!(env)
    env.reward = 0.0
    env.total_reward = 0.0
    env.actions = MinimalRLCore.get_actions(env, nothing)
    env.done = false
    return env.state
end

"""
    step!(env::GymEnv, a)

take a step in the enviroment
"""
function MinimalRLCore.environment_step!(env::GymEnv, a, args...)
    pyact = pyaction(a)
    pycall!(env.pystepres, env.pystep, PyObject, pyact)

    env.pystate, env.reward, env.done, env.info =
        convert(Tuple{PyObject, Float64, Bool, PyObject}, env.pystepres)

    convert_state!(env)

    env.total_reward += r
    return (r, env.state)
end

convert_state!(env::GymEnv{T}) where T =
    env.state = convert(T, env.pystate)

convert_state!(env::GymEnv{<:PyArray}) =
    env.state = PyArray(env.pystate)

# Reinforce.finished(env::GymEnv)     = env.done
# Reinforce.finished(env::GymEnv, s′) = env.done
MinimalRLCore.is_terminal(env::GymEnv) = env.done
MinimalRLCore.get_reward(env::GymEnv) = env.reward
MinimalRLCore.get_state(env::GymEnv) = env.state

# --------------------------------------------------------------

const pygym    = PyNULL()
const pysoccer = PyNULL()

function __init__()
    # the copy! puts the gym module into `pygym`, handling python ref-counting
    copy!(pygym, pyimport("gym"))
end

end # module
