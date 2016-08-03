# Julia code for generating RB pulse sequences for 1- and 2- qubit
# implementations of ORBIT.  A copy of the functionality of the
# orbit Mathematica notebook, but optimized for speed.  Shown at the end,
# a simple laptop can generate 10K length-15 pulse sequences for
# ORBIT on a CZ gate in under a second.


#============= Matrices =============#
# Clifford matrices can be normalized to have all entries one of
# {0, +1, -1, +i, -i}.  This is a 5-element set closed under multiplication,
# so we can just use the 5-element field.  It turns out addition is closed
# as well, up to rescaling.  A very fortunate isomorphism.

# The clifford matrix is 16 elements and all are part of the 5-element field,
# so in principle the entire matrix is one UInt64's worth of information.
# In practice, that kind of bit twiddling is likely slower than machine
# vectorized integer operations, so instead we will just use matrices
# of UInt8s, taking things mod 5 at the end of a calculation.

typealias CMatrix Matrix{UInt8}

# Our normalization convention is that the first nonzero entry be a 1.  This
# normalization reconciles the 5-element field and the +1, -1, +i, -i
# representations, showing both are isomorphic to the Clifford group.
function normalize(A::CMatrix)
  normConst = A[findfirst(A)]
  inverse = normConst $ ((normConst >>> 1) & 1) # Inverse mod 5 via bit twiddling..
  mod(A*inverse, 5)
end

# Implement multiplication such that the matrix product is properly normalized.
# Note that this function throws an error if called with no arguments, because
# it cannot infer what dimensions to use.
function mult(A::CMatrix...)
  normalize(reduce((B,C) -> mod(B*C, 5), eye(A[1]), A))
end

# We also have to modify matrix inversion.
function invert(A::CMatrix)
  normalize(map(x -> UInt8(round(mod(x, 5))), inv(A) * det(A)))
end

# At present, this function only works properly with 2x2 matrices as input.
# It outputs a 4x4 matrix corresponding to A on qubit 1 and B on qubit 2.
function outer(A::CMatrix, B::CMatrix)
  mod(hvcat((2,2), A[1]*B, A[3]*B, A[2]*B, A[4]*B), 5)
end

# I will explicitly write out the x/2, y/2 single-qubit matrices, and
# use these to generate the rest.  I will refer to the x/2 and y/2 as f and
# g, respectively.
const f = UInt8[1 2; 2 1]
const g = UInt8[1 4; 1 1]
const h = mult(g, f, g, g, g) # Gives the z/2 rotation

# For later use, the 2-qubit matrices cz and swap
const cz = UInt8[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 4]
const swap = UInt8[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]


# Single qubit clifford group is 24 gates, I'll just do this the long way.
# In the Mathematica notebook these are shown to correspond 1:1 with the pulses
# listed below (keepiing in mind that pulses are time-ordered left to right,
# while matrices are applied right to left)
const SQClif = CMatrix[eye(f),
               f,
               g,
               h,
               mult(f, f),
               mult(g, g),
               mult(h, h),
               mult(f, f, f),
               mult(g, g, g),
               mult(h, h, h),
               mult(f, g),
               mult(f, g, g, g),
               mult(f, f, f, g),
               mult(f, f, f, g, g, g),
               mult(f, h),
               mult(f, h, h, h),
               mult(f, f, f, h),
               mult(f, f, f, h, h, h),
               mult(f, f, g),
               mult(f, f, g, g, g),
               mult(g, g, h),
               mult(g, g, h, h, h),
               mult(h, h, f),
               mult(h, h, f, f, f)]


# 2-qubit clifford group can be constructed as a SQClif gate on each qubit,
# followed by one of the ten operations below, followed by an optional swap.
# We use only the "easy" half of the clifford group, modding out by swaps.
const EntanglingGate = CMatrix[eye(cz),
                       cz,
                       mult(outer(eye(f), f), cz),
                       mult(outer(eye(f), g), cz),
                       mult(outer(f, eye(f)), cz),
                       mult(outer(f, f), cz),
                       mult(outer(f, g), cz),
                       mult(outer(g, eye(f)), cz),
                       mult(outer(g, f), cz),
                       mult(outer(g, g), cz)]


#============= Pulses =============#

# We need an understandable way to designate the gates, so that an XY and Z control
# line can quickly convert between this program's output and pulse sequencing.  It
# would be best to not leave it tied to any specific number of qubits.
typealias Pulse Matrix{Int8}

const nul = Int8[[] []]  # Distinguish between null gate and idle gate
const I = Int8[0 0]
const Xpi2 = Int8[1 0]
const Xpi = Int8[2 0]
const X3pi2 = Int8[3 0]
const Ypi2 = Int8[4 0]
const Ypi = Int8[5 0]
const Y3pi2 = Int8[6 0]
const Zpi2 = Int8[0 1]
const Zpi = Int8[0 2]
const Z3pi2 = Int8[0 3]
const CZ = Int8[7 4 7 4]  # The idle on the XY line during a CZ is longer,
                      # so we give it a different signal from the idle gate

const SQPulse = Pulse[nul,
           Xpi2,
           Ypi2,
           Zpi2,
           Xpi,
           Ypi,
           Zpi,
           X3pi2,
           Y3pi2,
           Z3pi2,
           [Ypi2; Xpi2],
           [Y3pi2; Xpi2],
           [Ypi2; X3pi2],
           [Y3pi2; X3pi2],
           [Zpi2; Xpi2],
           [Z3pi2; Xpi2],
           [Zpi2; X3pi2],
           [Z3pi2; X3pi2],
           [Ypi2; Xpi],
           [Y3pi2; Xpi],
           [Zpi2; Ypi],
           [Z3pi2; Ypi],
           [Xpi2; Zpi],
           [X3pi2; Zpi]]

# With the above defined, we can make a lookup dictionary for the recovery pulse
SQLookup = Dict{CMatrix, Pulse}(map(=>, SQClif, SQPulse))

# And with only slightly more effort, we can do the same for the 2-qubit case.
# We need to list the pulses for the entangling gates:
const EntanglingPulse = Pulse[[nul nul],
                   CZ,
                   [CZ; [I Xpi2]],
                   [CZ; [I Ypi2]],
                   [CZ; [Xpi2 I]],
                   [CZ; [Xpi2 Xpi2]],
                   [CZ; [Xpi2 Ypi2]],
                   [CZ; [Ypi2 I]],
                   [CZ; [Ypi2 Xpi2]],
                   [CZ; [Ypi2 Ypi2]]]


# A robust way to cooordinate pulses of unknown lengths on two qubits in parallel.
function pulseOuter(A::Pulse, B::Pulse)
  if A == nul
    if B != nul
      A = I
    else
      return [A B]
    end
  elseif B == nul
    B = I
  end

  pad = size(A, 1) - size(B, 1)
  if pad == 1
    [A [B; I]]
  elseif pad == -1
    [[A; I] B]
  else
    [A B]
  end
end

# And then the analogous (5760-element) lists and lookup table.
const TQClif = collect([mult(EntanglingGate[k], outer(SQClif[i], SQClif[j]))
          for i=1:24, j=1:24, k=1:10])
const TQPulse = collect([[pulseOuter(SQPulse[i], SQPulse[j]); EntanglingPulse[k]]
          for i=1:24, j=1:24, k=1:10])

TQLookup = Dict{CMatrix, Pulse}(map(=>, [TQClif map(A->mult(swap, A), TQClif)]
	                                  , [TQPulse TQPulse]))


# From this we can build a function which chooses uniformly from the group,
# building sequences of nClifs operations ultimately ending in the identity.
# While we treat the 1-qubit and 2-qubit cases separately, it is mostly
# copy-paste.

# These functions take a gate to benchmark, given by its index in the SQClif or
# TQClif lists.  These functions are not meant to be called directly by users,
# who would not know this index.  Calling functions should find the index using
# find() or findfirst() and some more meaningful description of the gate, either
# the matrix in *Clif or the pulse sequence in *Pulse.  For a baseline measurement,
# the null gate should be used, which has empty lists as pulses.  This is index 1.

# This leads to a bit of complication if we want to benchmark the idle gate, which
# explicitly idles for a short time instead of proceeding directly to the next
# clifford.  One possibility is to add to the gateset an Idle pulse which has the
# same matrix but a different pulse and different index.  It would be left out
# of the lookup dictionary.

function benchmark1Qubit(pulseIndex::Int, nClifs)
  selection = rand(1:24, nClifs-1)
  recovery = invert(mult([SQClif[reverse(selection)]
  	                      fill(SQClif[pulseIndex], nClifs-1)]'[:]...))
  [vcat(SQPulse[[selection fill(pulseIndex, nClifs-1)]'[:]]...); SQLookup[recovery]]
end

function benchmark2Qubit(pulseIndex::Int, nClifs)
  selection = rand(1:5760, nClifs-1)
  recovery = invert(mult([TQClif[reverse(selection)]
  	                      fill(TQClif[pulseIndex], nClifs-1)]'[:]...))
  [vcat(TQPulse[[selection fill(pulseIndex, nClifs-1)]'[:]]...); TQLookup[recovery]]
end

#============= Fielity from RB =============#
# TODO FOR BETTER ORGANIZATION: break these sections into separate files

# These functions exist to determine the fidelity of a gate such that all
# error channels are mapped to a depolarizing channel.  The experiment consists
# of preparation in the 0 or 00 state, execution of these pulses, and measurement
# in the 0/1 basis on each qubit.  The error rate is the proportion of the
# measurements that return a 1, which can be due to the measurement itself or due
# to errors in the gates.  By having data for many numbers of gates we can
# separate the two sources and determine the gate fidelity.

# It comes down to an optimization algorithm which maps out the landscape
# of fidelity as a function of gate parameters. Here for an abstract gate we
# script the whole benchmarking process.  Some hardware communication functions
# still need to be written.

#=
function outcome1Qubit(pulseIndex::Int, sequenceLength::Int)
  sendPulses(benchmark1Qubit(pulseIndex, sequenceLength))
  measureAndZero()
end

# return a 1 or a 0 depending on the state of the qubit, and perform an Xpi pulse
# if the state was 1.  This leaves the qubit in the 0 state deterministically.
function measureAndZero()
  ret = readout()
  if (ret == 1)
  	sendPulses([Xpi])
  end
  ret
end
  
=#

# Placeholder code that assumes 90% measurement fidelity and 99.8% gate fidelity
function outcome1Qubit(pulseIndex::Int, nClifs::Int)
  Int(rand() < 0.9 * exp(-0.002*size(benchmark1Qubit(pulseIndex, nClifs), 1)))
end

function outcome2Qubit(pulseIndex::Int, nClifs::Int)
  Int(rand() < 0.9 * exp(-0.002*size(benchmark2Qubit(pulseIndex, nClifs), 1)))
end

# To completely eliminate systematic errors, I suppose we should test every
# sequence length randomly instead of in an order.  If our goal is 20000 data
# points each for 15 sequence lengths between 1 and 250, that is 300K pulse
# sequences that could be performed in any order so long as it was kept track of.

# I can't think of an argument against just doing a pulse sequence of each length,
# then repeating a sufficient amount of times to generate good statistics.  It
# prevents any variations in the fridge state over large times from interfering
# with results, and there should be no reason that the length of the previous
# pulse sequence should influence the fidelity of the following one.  If we want
# to eliminate that influence we could scramble the order each time, but the
# overhead might not be worth it.

function fidelity1Qubit(gate::Pulse, averaging::Int, sequenceLengths)
  pulseIndex = findfirst(SQPulse, gate)
  # measureAndZero()
  baseline = [outcome1Qubit(1, nClifs)
      for nClifs in sequenceLengths, i in 1:averaging]'
  rawData = [outcome1Qubit(pulseIndex, nClifs)
      for nClifs in sequenceLengths, i in 1:averaging]'
  targetOnly = [mean(rawData[:,i]) / mean(baseline[:,i])
      for i in 1:length(sequenceLengths)]
  seqFidelity = 1 + getSlope(sequenceLengths, map(log, targetOnly))
end

function fidelity2Qubit(gate::Pulse, averaging::Int, sequenceLengths)
  pulseIndex = findfirst(TQPulse, gate) 
  # measureAndZero()
  baseline = [outcome2Qubit(1, nClifs)
      for nClifs in sequenceLengths, i in 1:averaging]'
  rawData = [outcome2Qubit(pulseIndex, nClifs)
      for nClifs in sequenceLengths, i in 1:averaging]'
  targetOnly = [mean(rawData[:,i]) / mean(baseline[:,i])
      for i in 1:length(sequenceLengths)]
  seqFidelity = 1 + getSlope(sequenceLengths, map(log, targetOnly))
end

# basic linear regression
function getSlope(xs, ys)
  cx = xs - mean(xs)
  cy = ys - mean(ys)
  dot(cx, cy) / dot(cx, cx)
end

# For ORBIT, we need something that can be generated more quickly.  Rather than
# try to extract the gate fidelity from measurements of multiple sequence lengths,
# we just pick an appropriately long sequence and optimize its average fidelity,
# no baseline or other modifications.  Upon varying the pulse parameters, the
# sequence fidelity is expected to change, so we can maximize it.
function orbitFitness(gate::Pulse, averaging::Int, sequenceLength::Int)
  pulseIndex = findfirst(SQPulse, gate)
  mean([outcome1Qubit(pulseIndex, sequenceLength) for i in 1:averaging])
end


#============= Optimization =============#
# For the optimization algorithm, I was wondering if a particle-swarm stochastic
# algorithm would outperform those employed by other active groups.  The advantage
# is that stochastic algorithms have a better expected performance on spaces with
# many parameters (I've seen PSO employed successfully on a 16-dimensional space
# with very narrow peaks in the fitness function).  With freedom to define many more
# parameters than previous groups, we could even try to find a meshing of the
# spectrum of the pulse and tune every point in that mesh independently.

# Explicitly, if we have N 14-bit digital points, converted to an analog signal
# with an appropriate low-pass filter, we could choose to optimize all N points.
# Or we could find the fourier components of the pulse and optimize them.  This
# has to be done subject to the constraints that the voltage is zero at T = 0 and
# T = pulselength, which I will assume is 20ns.

# Looking on GitHub, the only PSO library built in Julia has precisely zero
# comments, and the "Usage" section of the Readme is left blank.  So I don't know
# if we can rely on it.  There is a MATLAB library implementing it but I might just
# prefer to roll our own, tailored to waveform shaping.

# A PSO solution is a list of the values for all the parameters of interest.  For us,
# it could just be every point in the waveform.  Its fitness is determined by sending
# that pulse to the AWG as the definition of a gate, and benchmarking that gate using
# orbitFitness defined above.  For good SNR we need high averaging, so a single
# fitness measurement could take a couple milliseconds.  Keep this in mind when
# comparing the performance to the pure-math fitness function below, which likely
# takes the computer less than a microsecond.

#=
function fitness(waveform::Array{Any,1})
  sendToAWG(waveform, otherArgs)
  orbitFitness(args)
end
=#

# Let's momentarily use a fitness function that seeks out a 16-bit digitized
# sine wave on 20 points, separated by 1 radian apiece (so 2pi points per cycle).
# This is just a test of how many iterations it takes to converge to a waveform.
# With settings 
# popSize = 1000
# selfWeight = neighborWeight = 1
# neighborhoodMin = 10
# inertiaMin = 0.007
# inertiaMax = 0.5
# we get convergence to the global optimum in 75 iterations
#
# With the same parameters but adjusting the fitness to seek out a 200-point function
# rather than a 20-point function, we converge in about 350 iterations.
function fitness(params::Array{Int,1})
  return -log(sum([(params[i]-16384*(1+sin(i)))^2
  	                               for i in 1:length(params)]))
end

type Particle
	position
	currentFitness
	velocity
	bestPosition
	bestFitness
end

# Permit construction with no history
Particle(x, f, v) = Particle(x, f, v, x, f)
# Permit construction with unknown fitness
Particle(x, v) = Particle(x, fitness(x), v)

# Compare two particles.  Julia can find maximum element of an array so long as
# Base.isless is defined.
Base.isless(x::Particle, y::Particle) = x.currentFitness < y.currentFitness

# Implement PSO where position is a vector of integers.
# Still need to find sensible default values for our application so the args list
# isn't overwhelming and confusing every time.
function psoSkeleton(popSize::Int,
	        boundsMin::Array{Int,1}, boundsMax::Array{Int,1},
	        selfWeight, neighborWeight,
	        neighborhoodMin::Int,
	        inertiaMin, inertiaMax,
	        maxIterations::Int)

  # Initialize a pool of "Popsize" Particles, with each element in the bounds
  startingPos = hcat(map((x,y) -> rand(x:y, popSize), boundsMin, boundsMax)...)'
  startingVel = hcat(map(x -> rand(-x:x, popSize), boundsMax - boundsMin)...)'
  populationInfo = [Particle(startingPos[:,i], startingVel[:,i])
                    for i in 1:popSize]
  winner = findmax(populationInfo)[1]

  # Initialize our running PSO variables
  (winnerX, winnerF) = (winner.position, winner.currentFitness)
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
  
    # New velocities are a weighted sum of old velocity, distance to local winner
    # and distance from personal best
    map((x,y) -> x.velocity = W*x.velocity +
    	       selfWeight*rand(length(x.position)).*(x.bestPosition - x.position)
             + neighborWeight*rand(length(x.position)).*(y.position - x.position),
                 populationInfo, localWinners)
  
    # Update the positions based on these new velocities (can you tell I prefer
    # functional programming?).  The bit with abs is to clip it to the proper range.
    map(x -> x.position = map((p,lo,hi) ->
    	  div(lo + hi + abs(Int(round(p))-lo) - abs(Int(round(p))-hi), 2),
    	  x.position + x.velocity, boundsMin, boundsMax), populationInfo)
  
    # Update the current fitness.  If it is better than the old fitness, save the
    # current position.  If it is the best seen so far, udpate best fitness.  This
    # one will be done sequentially since the fitness calls can't be parallelized.
    for x in populationInfo
        x.currentFitness = fitness(x.position)
        if (x.currentFitness > x.bestFitness)
        	x.bestPosition = x.position
        	x.bestFitness = x.currentFitness
        end
        if (x.currentFitness > winnerF)
        	(winnerX, winnerF) = (x.position, x.currentFitness)
        	improvementFlag = true
        end
    end
  
    # Update PSO variables depending on state
    if improvementFlag
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
  println(winnerF)

  iters += 1
  end # End loop 

  # Return our best found position and its fitness score
  (winnerX, winnerF)

end

# I don't like functions that take more than a screen..  But I'm happy to leave
# this one without any additional helpers, I think.  At least for now.