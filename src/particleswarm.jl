#======================= Particle Swarm Optimization ==========================#

# Particle Swarm Optimization is a stochastic optimization algorithm without
# interbreeding.  It mimics a flock of birds, who prefer to spread out but will
# hone in if one finds what they are all looking for (in a very vague sense).

# A PSO solution is a list of the values for all the parameters of interest.
# These parameters must lie within user-specified bounds.  The goal is to
# find the set of parameters which maximizes a "fitness" function.  PSO is
# typically used to maximize erratic functions which depend on a large number of
# parameters.  (My introduction to PSO was http://boxcar2d.com/, where one
# optimizes the performance of a polygonal car with over 16 degrees of freedom.
# Generic algorithm and PSO are both options - PSO was better at fine-tuning)

module ParticleSwarm
export PSO

type Particle{T<:Vector}
  position::T
  currentFitness
  velocity::T
  bestPosition::T
  bestFitness
end

# Permit construction with no history
Particle{T<:Vector}(x::T, f, v::T) = Particle{T}(x, f, v, x, f)

# Compare two particles.  Julia can find maximum element of an array so long as
# Base.isless is defined.
import Base.isless
Base.isless(x::Particle, y::Particle) = x.currentFitness < y.currentFitness

# Helper to generate <amount> random numbers between two bounds.  Only uses
# Float64 precision but that should be enough for us.
randBetween(min::Float64, max::Float64, amount=1) = min + (max - min)*rand(amount)
randBetween(min::Integer, max::Integer, amount=1) = rand(min:max, amount)

# The simplest arguments to a generic PSO function are the population size, the
# fitness function, and the bounds on the parameters.  As solution candidates
# follow swarm mechanics, a sense of "inertia" is present - alterations happen
# to the velocity rather than directly to the position.  How many neighbors can
# influence this velocity varies depending on whether there has been recent
# success.  If the swarm is consistently improving the fitness every iteration,
# all candidates will follow closely.  If many iterations have gone by with no
# improvement, the candidates follow their own dispersion rather than take input
# from their neighbors.  Lastly, the user must specify how many iterations to
# run before returning the best solution found.

# Our PSO requires all parameters be real valued so that there is a sense of
# being between bounds.  It could be better generalized to anything comparable,
# but I am using functions that assume real numbers.

function PSO{T<:Real}(
          fitness::Function,
          boundsMin::Vector{T}, boundsMax::Vector{T};
          popSize::Integer=0, # If not overridden, defaults to 5x number of vars
          neighborhoodMin::Integer=0, # If not overridden, defaults to 1/10 pop
          maxIterations::Integer=1000,
          inertiaMin=0.007, inertiaMax=0.5,
          selfWeight=1, neighborWeight=1,
          candidate::Vector{T}=randBetween.(boundsMin, boundsMax))

  popSize = (popSize > 0) ? popSize : 3*length(boundsMin)
  neighborhoodMin = (neighborhoodMin > 0) ? neighborhoodMin : div(popSize, 10)

  # Initialize a pool of "Popsize" Particles, with each element in the bounds
  pinit = [candidate'; hcat(randBetween.(boundsMin, boundsMax, popSize-1)...)]
  vinit = hcat(randBetween.(boundsMin-boundsMax, boundsMax-boundsMin, popSize)...)
  finit = fitness([pinit[i,:] for i in 1:popSize])
  populationInfo = [Particle(pinit[i,:], finit[i], vinit[i,:])
                    for i in 1:popSize]
  winner = findmax(populationInfo)[1]

  # Initialize our running PSO variables
  winnerX, winnerF = winner.position, winner.currentFitness
  N = neighborhoodMin
  W = inertiaMax
  stallCounter = 0
  iters = 0

  # An iteration.
  while(iters < maxIterations && stallCounter < 100)
    improvementFlag = false

    # For each element find a subset of length N not including
    # the element itself and identify the winner.
    neighborhoods = [map(x -> x + Int(x >= i), randperm(popSize-1)[1:N])
                      for i in 1:popSize]
    localWinners = map(n -> findmax(populationInfo[n])[1], neighborhoods)

    # New velocities are a weighted sum of old velocity, distance to local
    # winner and distance from personal best
    map((x,y) -> x.velocity = round.(W*x.velocity +
           selfWeight*rand(length(x.position)).*(x.bestPosition - x.position)
         + neighborWeight*rand(length(x.position)).*(y.position - x.position)
         , populationInfo, localWinners))

    # Update the positions based on these new velocities (can you tell I prefer
    # functional programming?). We clip it to the proper range.
    map(x -> x.position = map((p,lo,hi) -> div(lo + hi + abs(p-lo) - abs(p-hi),2),
        x.position + x.velocity, boundsMin, boundsMax), populationInfo)

    # Update the current fitness.  If it is better than the old fitness, save
    # current position.  If it is the best seen so far, udpate best fitness.
    # Because we have a noisy signal, we also re-check the fitness of our
    # global best, and for better or worse assume the new value.
    fitnesses = fitness([[winnerX]; (x->x.position).(populationInfo)])
    maxF, ind = findmax(fitnesses)
    improvementFlag = (ind > 1)
    winnerF = maxF
    for n in 1:popSize
        x = populationInfo[n]
        x.currentFitness = fitnesses[n + 1]
        if (x.currentFitness > x.bestFitness)
          x.bestPosition = x.position
          x.bestFitness = x.currentFitness
        end
    end

    # Update PSO variables depending on state
    if improvementFlag
      winnerX = populationInfo[ind - 1].position
      stallCounter = max(0, stallCounter-1)
      N = neighborhoodMin
    else
        stallCounter += 1
      N = min(N + neighborhoodMin, popSize - 1)
    end

    if stallCounter < 2
      W = min(2*W, inertiaMax)
    elseif stallCounter > 5
      W = max(W/2, inertiaMin)
    end

  # Debug output
  println(iters)
  poolF, poolInd = findmax(fitnesses[2:(1+popSize)])
  println("Highest fitness this generation: "*string(poolF)*" by member "*string(poolInd - 1))
  println("Global optimum is "*string(winnerF)*" by parameters "*string(winnerX))


  iters += 1
  end # End loop

  # Return our best found position and its fitness score
  winnerX, winnerF
end

end # End module ParticleSwarm
