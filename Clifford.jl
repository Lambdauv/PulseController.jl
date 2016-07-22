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

const nul = [[] []]  # Distinguish between null gate and idle gate
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
# This is essentially 4 copies of the same function, two for single qubit and
# two for 2 qubits, with one each for a baseline and one each interleaving a
# user-specified gate.

# Example: benchmark2Qubit(CZ, 11) generates a pulse sequence of 10 random 2-qubit
# cliffords interleaved with 10 CZ gates, followed by an 11th recovery clifford.

# To benchmark a single-qubit gate in a 2-qubit system, the syntax uses [].
# benchmark2Qubit([Xpi2 I], 15) benchmarks a pi/2 X pulse on qubit 1.

function baseline1Qubit(nClifs)
  selection = rand(1:24, nClifs-1)
  recovery = invert(mult(SQClif[reverse(selection)]...))
  [vcat(SQPulse[selection]...); SQLookup[recovery]]
end

function baseline2Qubit(nClifs)
  selection = rand(1:5760, nClifs-1)
  recovery = invert(mult(TQClif[reverse(selection)]...))
  [vcat(TQPulse[selection]...); TQLookup[recovery]]
end

function benchmark1Qubit(gate::Pulse, nClifs)
  selection = rand(1:24, nClifs-1)
  pulseIndex = findfirst(SQPulse, gate)
  recovery = invert(mult([SQClif[reverse(selection)]
  	                      fill(SQClif[pulseIndex], nClifs-1)]'[:]...))
  [vcat(SQPulse[[selection fill(pulseIndex, nClifs-1)]'[:]]...); SQLookup[recovery]]
end

function benchmark2Qubit(gate::Pulse, nClifs)
  selection = rand(1:5760, nClifs-1)
  pulseIndex = findfirst(TQPulse, gate)
  recovery = invert(mult([TQClif[reverse(selection)]
  	                      fill(TQClif[pulseIndex], nClifs-1)]'[:]...))
  [vcat(TQPulse[[selection fill(pulseIndex, nClifs-1)]'[:]]...); TQLookup[recovery]]
end

# A function to see how quickly this code can generate clifford pulse sequences
function metabenchmarking(ntrials, gate::Pulse, nClifs)
  for i in 1:ntrials
    benchmark2Qubit(gate, nClifs) # Throw away output
  end
end

# My output (ymmv)
# > @time metabenchmarking(10000, CZ, 15)
#     0.946772 seconds (4.17 M allocations: 262.577 MB, 2.67% gc time)