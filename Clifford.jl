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
  inverse = normConst $ ((normConst >>> 1) & 1) # Inverse mod 5
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

# Output a clifford matrix corresponding to A on qubit 1 and B on qubit 2.
function outer(A::CMatrix, B::CMatrix)
  mod(kron(A, B), 5)
end

# I'll explicitly write out the x pi/2 and y pi/2 single-qubit matrices, using
# these to generate the rest.  I will refer to them as f and g, respectively.
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

# We need an understandable way to designate the gates, so that control lines
# can quickly convert between this program's output and pulse sequencing.  It
# would be best to not leave it tied to any specific number of qubits.
typealias Pulse Matrix{Int8}

# There are two potentially desirable ways to perform clifford gates.  One is
# in the fewest number of overall gates, assuming XY and Z lines are present.
# The other is in the fewest number of XY gates.  If ZLine is set to true,
# the program will be able to do RB seqeunces with the same effect in fewer
# gates, and the Qubit type will ask for the channel for all 3 lines: XYI, XYQ
# and Z.  If ZLine is set to false, the gates performed will only exist on the
# XYI and XYQ lines, and the Qubit type will not ask about a Z line.  The latter
# is necessary for doing single qubit RB with only the AWG, since there are 4
# lines and 2 are used for readout.
const ZLine = false

# This is essentially an enum, but the numbers are chosen to have some later
# significance.  Note the null gate is empty rather than idle.
const nul = Array(Int8,(0,1))
const I =     fill(Int8(7),(1,1))
const Xpi2 =  fill(Int8(1),(1,1))
const Xpi =   fill(Int8(5),(1,1))
const X3pi2 = fill(Int8(3),(1,1))
const Ypi2 =  fill(Int8(2),(1,1))
const Ypi =   fill(Int8(6),(1,1))
const Y3pi2 = fill(Int8(4),(1,1))
const Zpi2 =  fill(Int8(8),(1,1))
const Zpi =   fill(Int8(9),(1,1))
const Z3pi2 = fill(Int8(10),(1,1))
const CZ =    Int8[15 15]  # CZ requires 2 qubits to implement.

const SQPulse = (ZLine ?
Pulse[nul,
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
:
Pulse[nul,
      Xpi2,
      Ypi2,
      [Xpi2; Ypi2; X3pi2],
      Xpi,
      Ypi,
      [Xpi; Ypi],
      X3pi2,
      Y3pi2,
      [Xpi2; Y3pi2; X3pi2],
      [Ypi2; Xpi2],
      [Y3pi2; Xpi2],
      [Ypi2; X3pi2],
      [Y3pi2; X3pi2],
      [Xpi2; Ypi2],
      [Xpi2; Y3pi2],
      [X3pi2; Y3pi2],
      [X3pi2; Ypi2],
      [Ypi2; Xpi],
      [Y3pi2; Xpi],
      [Y3pi2; Xpi2; Y3pi2],
      [Ypi2; Xpi2; Ypi2],
      [Ypi; Xpi2],
      [Ypi; X3pi2]]
)

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


# Coordinate pulses of unknown lengths on two qubits in parallel.  This is in
# complement to "outer" defined above, in that if pulse A has matrix mA, B has
# matrix mB, then pulseOuter(A, B) has matrix given by outer(mA, mB).
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
  if pad == 2
    [A [B; I; I]]
  elseif pad == 1
    [A [B; I]]
  elseif pad == -1
    [[A; I] B]
  elseif pad == -2
    [[A; I; I] B]
  else
    [A B]
  end
end

# And then the analogous (5760-element) lists and lookup table.
const TQClif = collect([mult(EntanglingGate[k], outer(SQClif[i], SQClif[j]))
          for i=1:24, j=1:24, k=1:10])
const TQPulse= collect([[pulseOuter(SQPulse[i], SQPulse[j]); EntanglingPulse[k]]
          for i=1:24, j=1:24, k=1:10])

# The lookup table finds the pulse corresponding to either the matrix given or
# that matrix plus a swap operation.  Since we use this lookup table to reach
# the ground state, both will work.  We will not be turning error states into
# clean ones, just switching which qubit fails to read ground state.
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
# the matrix in *Clif or the pulse sequence in *Pulse.  For a baseline, the null
# gate should be used, which has empty lists as pulses.  This is index 1.

# This leads to a bit of complication if we want to benchmark the idle gate,
# which explicitly idles for a short time instead of proceeding directly to the
# next clifford.  One possibility is to add to the gateset an Idle pulse which
# has the same matrix but a different pulse and different index.  It would be
# left out of the lookup dictionary.  TODO: add support for benchmarking "idle."

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
# of preparation in the 0 or 00 state, execution of RB pulses, and measurement
# in the 0/1 basis on each qubit.  The error rate is the proportion of the
# measurements that return a 1, which can be due to the measurement itself or
# errors in the gates.  By having data for different seqeunce lengths we can
# separate the two sources and determine the gate fidelity.

# It comes down to an optimization algorithm which maps out the landscape
# of fidelity as a function of gate parameters. Here for an abstract gate we
# script the whole benchmarking process.  Some hardware communication functions
# still need to be written.


# Placeholder code that assumes 90% measurement fidelity and 99.8% gate fidelity
function outcome1Qubit(pulseIndex::Int, nClifs::Int)
  Int(rand() < 0.9 * exp(-0.002*size(benchmark1Qubit(pulseIndex, nClifs), 1)))
end

function outcome2Qubit(pulseIndex::Int, nClifs::Int)
  Int(rand() < 0.9 * exp(-0.002*size(benchmark2Qubit(pulseIndex, nClifs), 1)))
end

# To completely eliminate systematic errors, we won't do many consecutive
# seqeunces of the same length for one data point, then a new length.  Fridge
# instability, flux noise, and more could make time a legitimate factor in
# such an experiment.  Instead, if our goal is, say, 20000 data points each for
# 15 sequence lengths between 1 and 250, that is 300K pulse sequences that
# could be performed in any order so long as it was kept track of.

# I can't think of an argument against just doing a pulse sequence of each
# length, then repeating a sufficient amount of times for averaging.  It
# prevents any variations in the fridge state over large times from interfering
# with results, and there should be no reason that the length of the previous
# pulse sequence should influence the fidelity of the following one.  The only
# issue is if the set of all seqeunce lengths took time equal to some
# oscillatory noise source, like the pulse tube..  If we want to eliminate that
# influence we could scramble the order each time, but the overhead might not
# be worth it.
function fidelity1Qubit(gate::Pulse, averaging::Int, sequenceLengths)
  pulseIndex = findfirst(SQPulse, gate)
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

const orbitSequenceLength = 15
const orbitAveraging = 5000

# For optimization algorithms, we need something that can be generated more
# quickly.  Rather than try to extract the gate fidelity from measurements of
# multiple sequence lengths, we can just pick an appropriately long sequence
# and optimize its average fidelity, which includes the error of the cliffords
# and of the measurement.  Upon varying the pulse parameters, the sequence
# fidelity changes in 1:1 correspondence with that of the gate, just with
# higher visibility and more easily acquired.




# Let's plan out which steps of optimization take place on which instruments,
# assuming FPGAs and 2 qubits (near future).  The computer is in charge of
# determining gate lists with parallelism and recovery, and can do so.  The FPGA
# is in charge of taking the resulting gate lists and making waveform sequences.
# The XY lines need an IF carrier and an envelope; the carrier provides full I/Q
# data and the envelope the frequency broadening to get an adiabatic and 2-state
# safe pulse shape.  For the optimization, the envelope is what will vary.
# I will assume that loading a pulse takes a lot of time, so that the fastest
# way to do things is to load pulses as rarely as possible.  The FPGA should be
# able to read in a short sequence of integers from the control computer and
# from there piece together a sequence of preloaded pulses.  The computer
# will make sure that the pulse sequences requested are the same length on all
# channels.  In this, the FPGA is just a dummy that turns small amounts of
# input into the desired output very quickly.

# However, in sequences like ORBIT when the pulses being generated are
# subject to change, we will need to load many different pulses to the FPGA.
# If the number of pulses accepted by the device is enough, we could load a
# whole pool of options every iteration and then calculate the statistics for
# each.  The index sent by the control program for that specific gate would
# depend on which pulse from within the pool was acting.  It could be a good
# performance improvement to tailor sequencing code to allow multiple switchable
# definitions of the same gate.

# The other thing needed from the FPGA is a control loop to deterministically
# put the qubits in their zero state.  The loop would be measure - pulse if 1,
# repeat.  The first measurement would be returned to the control computer,
# and the pulse to do would be sent along with the command.  As described,
# it works perfectly for both ORBIT and error correction.  However, the QEC
# procedure actually conditions a single pulse on the outcome of multiple
# measurements, and the pulse would not be on the same qubit measured.  So
# there would likely be some transfer latency in this, and if that is the
# case, it would suffice to instead just wait a few T1 and assume ground
# state initialization.  That just takes a long time.

# The controller cares about which qubit and which line is being addressed.
# Presently readout is on all qubits on the chip, as they share the readout
# resonator, so readout is a separate issue.  For XY pulses, two lines must
# be active, sending I and Q data to the mixer.  The Z line sends the waveform
# directly to the qubit.

# Let's first define some constants.  The offsetValue corresponds to sending
# zero through the 14-bit AWG.  Not sure if the FPGA will have the same or
# different.  floatIdleLength is the gate length of an idle pulse.  On lines
# other than the current active line, say the Z line during an X pi pulse,
# we need a determined length to wait.  It is easiest to be consistent and
# always use the same pulse length if we are relying on idles for timing.

const offsetValue = 0x1fff
const floatIdleLength = 250 # If using the AWG, this should be 250.
const sampleRate = 1e9

# NOT IMPLEMENTED: "idles" holds the pulse index for each board corresponding to
# simply delaying by floatIdleLength samples.  Accordingly, with more boards
# we would need a longer list.  Not sure if "const" is appropriate.  The
# pretendIndexCounter is a placeholder to simulate the behavior of the FPGA,
# assigning new waveforms to the first available place in its memory.

awg = InsAWG5014C() # Takes a visession argument?
idles = Dict(awg => "I") # For AWG, indices are strings.
pretendIndexCounter = 0

#========================== The Waveform DataTypes ============================#
# There are two approaches we could take for storing waveforms.

# 1) A fast approach with assumptions.
#       In this, each named gate pulse just has one associated real waveform.
#       If it is an X or Y gate, the waveform is the window on a specific-
#       frequency IQ wave, and the Z is off during that pulse.  If it is a Z
#       gate, then the waveform is the signal sent to the Z line, and the XY
#       line is off during that pulse.  Due to the fact that XY signals must
#       be mutiplied with the IQ wave before rounding and sending to the DAC,
#       it is only sensible to work with real-valued waveforms in this approach.
#       In this case, all pulses must be the same duration, so that sequencing
#       code can splice in a generic idle pulse for the inactive lines.

# 2) A full approach with no assumptions but more data to optimize
#       In this, each named gate pulse has 3 associated integer waveforms,
#       designating the XYI, XYQ, and Z pulse shapes.  This allows for tactics
#       such as DRAG.  It is also "closer to the machine," in that the data
#       being held is exactly the digital data sent to the DAC, rather than
#       needing to be altered in some way beforehand.  For sequencing, all
#       lines draw from the waveform data for the gate at hand, meaning all
#       gates can nominally be independent lengths.

# In the event we are controlling a qubit with XY only, we need to adapt the
# ExactWaveform type to hold XYI and XYQ but not Z.  This will consist of length
# matched pulses on XYI and XYQ and an empty pulse on Z.  The lengths do not
# need to match the idle length, because the Z line simply doesn't exist.  Due
# to this necessity, I have removed the strict length-matching conditions on
# newly created ExactWaveform objects.  TODO: Find consistent set of checks
# for errors in initialization.  I don't want users sending pulses out of phase.

# When doing PSO and other optimization, it seems sensible to begin with the
# first approach to get a good gate, and then tune it up further with the second
# to get a by-point optimized gate.  Using Julia typing, we can make the switch
# painless.

# Pulses (equivalently, gates) live independently of which qubits they are on,
# so it seems the responsibility of the qubit type to maintain a dictionary of
# pulse => waveform, with waveform one of the two above collections of data.
# We can maintain the pulse typealias from above, which plays very nicely with
# sequencing (a sequence of multiple pulses is the same type as a single pulse),
# and this potentially gives us the ability to add to the qubit's dictionary any
# optimized waveform for arbitrary pulse sequences.

abstract Waveform

type FloatWaveform
  wavedata::Vector{Float64}
end

type ExactWaveform <: Waveform
  XYI::Vector{UInt16}
  XYQ::Vector{UInt16}
  Z::Vector{UInt16}
  dirty::Bool # Denotes whether this is up to date with the DAC's memory

  # Provide an inner constructor.  There are two options for an ExactWaveform.
  # If it encodes a FloatWaveform's data, it will have empty pulses on the
  # inactive lines.  All active lines must be length matched to the idle pulse.
  # If not, the only constraint is that all lines must be length matched to
  # one another.  Hence, 3 lines of comparators.  Julia allows x == y == z.

  # 9/9/16 - disabled constructor to accomodate exact XY-only control.
#=  ExactWaveform(XYI, XYQ, Z, dirty) =
    (length(XYI) == length(XYQ) == length(Z) ||
     length(XYI) == length(XYQ) == floatIdleLength && length(Z) == 0 ||
     length(XYI) == length(XYQ) == 0 && length(Z) == floatIdleLength) ?
    new(XYI, XYQ, Z, dirty) :
    error("Incompatible waveform lengths") =#
end

# Make ExactWaveforms readable
import Base.string
function string(x::ExactWaveform)
  str = x.dirty ? "Unsynced" : "Synced"
  str *= " waveform with the following pulse shapes:\nXYI: "
  str *= length(x.XYI) > 0 ? string([Int(c) for c in x.XYI]) : "idle"
  str *= "\nXYQ: "
  str *= length(x.XYQ) > 0 ? string([Int(c) for c in x.XYQ]) : "idle"
  str *= "\nZ: "
  str *= length(x.Z) > 0 ? string([Int(c) for c in x.Z]) : "idle/undefined"
end

import Base.print
print(io::IO, x::ExactWaveform) = print(io, string(x))

import Base.show
show(io::IO, x::ExactWaveform) = print(io, x)


#========================== The Qubit DataType ================================#
# For the purposes of this code, the only info needed about a qubit is its
# resonant frequency relative to the local oscillator, which boards/lines
# are able to communicate with it, and the on-board data for gates.

# Depending on if lines permit, we can control our qubits with only XY control
# or with full XYZ.  If the Z line is absent, the underlying type is a QubitNoZ,
# and if present it is a QubitWithZ.  The user only creates "Qubit" objects,
# and depending on whether a Z line is specified, the program generates the
# proper concrete type.

abstract Qubit

importall InstrumentControl
# Later, we will import other applicable control instrument types, like FPGAs

type QubitWithZ <: Qubit
  IFreq::Float64
  lineXYI::Tuple{Instrument,Int} # (board, channel)
  lineXYQ::Tuple{Instrument,Int} #  "
  lineZ::Tuple{Instrument,Int}   #  "
  pulseConvert::Matrix        # 10x3 matrix, which maps the 10 basic pulses
                              # to how the DAC's memory refers to them

  waveforms::Dict{Pulse, ExactWaveform} # See below.
end

type QubitNoZ <: Qubit
  IFreq::Float64
  lineXYI::Tuple{Instrument,Int} # (board, channel)
  lineXYQ::Tuple{Instrument,Int} #  "
  pulseConvert::Matrix        # 7x2 mapping basic pulses to XYI and XYQ labels
                              # in the DAC's memory (for AWG, this is strings,
                              # for reasons that only frustrate me..)

  waveforms::Dict{Pulse, ExactWaveform} # See below.
end

# Accessing a qubit as a dictionary will alter the contained dictionary of
# pulses.  Though the dictionary itself only contains ExactWaveform values,
# the methods below provide ways to convert FloatWaveform and Vector{Float64}
# values to ExactWaveform.  An explicit convert method cannot be written, since
# the ExactWaveform produced depends on the Pulse and on the IFreq of the qubit.
import Base.setindex!
function setindex!(q::Qubit, w::Vector{Float64}, p::Pulse)
  if p[1] == 7
    q.waveforms[p] = ExactWaveform(UInt16[], UInt16[], UInt16[], true)
  elseif length(w) != floatIdleLength
      error("FloatWaveform pulses must contain exactly "*
                                 string(floatIdleLength)*" points")
  elseif p[1] < 7
    (xyi, xyq) = IQgen(q.IFreq, p, w)
    q.waveforms[p] = ExactWaveform(xyi, xyq, UInt16[], true)
  elseif p[1] < 11 && isa(q, QubitWithZ)
    q.waveforms[p] = ExactWaveform(UInt16[], UInt16[], 
                    map(x -> UInt16(offsetValue + round(x)), w), true)
  else # Trying to set Z gates on a QubitNoZ object
    println("Warning: no gates set by this operation.")
  end
end

# A helper method for the above that uses the IFreq and the intended phase to
# determine the I and Q pulses.
function IQgen(IFreq::Float64, pulse::Pulse, window::Vector{Float64})
  if pulse[1] > 6
    error("Not a pulse in need of IQ mixing")
  else
    phase = im ^ pulse[1]
    helix = phase * [exp(im * 2π * IFreq * p / sampleRate) * window[p]
                                  for p in 1:length(window)]
    waveformI = map(z->UInt16(offsetValue + round(real(z))), helix)
    waveformQ = map(z->UInt16(offsetValue + round(imag(z))), helix)
    waveformI, waveformQ
  end
end

# For FloatWaveforms, use the contained wavedata field.
function setindex!(q::Qubit, w::FloatWaveform, p::Pulse)
  setindex!(q, w.wavedata, p)
end

# For ExactWaveforms, use them as they are.
function setindex!(q::Qubit, w::ExactWaveform, p::Pulse)
  if p[1] < 7 || isa(QubitWithZ)
    w.dirty = true
    q.waveforms[p] = w
  else # Trying to set Z gates on a QubitNoZ object
    println("Warning: no gates set by this operation.")
  end
end

# Always display the ExactWaveform for the user to see.
import Base.getindex
function getindex(q::Qubit, p::Pulse)
  return q.waveforms[p]
end

# ================= Initializing a Qubit's information ========================#

# A user shouldn't have to specify the matrix or the dictionary to define a
# qubit.  All that they should have to input is IFreq and the line info.
# A user can specify "False" for the Z line to generate a QubitNoZ object,
# or omit it entirely.
function Qubit(IFreq::Float64, lineXYI::Tuple{Int,Int}, lineXYQ::Tuple{Int,Int},
               lineZ=false)
  if isa(lineZ, Bool) && !lineZ
    ret = QubitNoZ(IFreq, lineXYI, lineXYQ, fill(-1, (7,2)), Dict())
  elseif isa(lineZ, Tuple)
    ret = QubitWithZ(IFreq, lineXYI, lineXYQ, lineZ, fill(-1, (10,3)), Dict())
  end
  println("To initialize pulse shapes for this Qubit, please run one of the "*
           "init routines:\n\tgaussInit\n\tcosInit\n\tgeneralInit")
  ret
end

# Initializing the dictionary shouldn't require manually inputting up to 10
# different pulse shapes, so we provide quick ways to make gaussian-envelope and
# cos-envelope pulses.  These are all FloatWaveform objects with all the
# assumptions mentioned above.  We also provide a generalInit function that
# takes a shape for the window, though it must be floatIdleLength points exactly
# or an error will be thrown down the line.

# These functions assume that half-amplitude pulses provide half the rotation.
# This is a good starting point but is not accurate for high-fidelity gates.

# The XY and Z arguments specify whether the pulses created are for the XY
# control or the Z control.  By default, both are on, but it is likely the
# ideal amplitude for each is different.  For QubitNoZ objects, setting Z to
# true or false makes no difference.
function gaussInit(q::Qubit, amplitude, sigma, XY::Bool = true, Z::Bool = true)
  pulseShape = [amplitude*exp(-(x - (floatIdleLength + 1)/2)^2 / (2*sigma^2))
                                for x in 1:floatIdleLength]
  pulseShape -= pulseShape[1] # To eliminate the tails
  generalInit(q, pulseShape, XY, Z)
end

function cosInit(q::Qubit, amplitude, XY::Bool = true, Z::Bool = true)
  pulseShape = [amplitude*(1-cos(x/floatIdleLength)/2)
                                for x in 1:floatIdleLength]
  generalInit(q, pulseShape, XY, Z)
end

function generalInit(q::Qubit, pulseShape, XY::Bool = true, Z::Bool = true)
  if XY
    q[Xpi] = q[Ypi] = pulseShape
    q[Xpi2] = q[Ypi2] = q[X3pi2] = q[Y3pi2] = pulseShape/2
  end
  if Z && isa(q, QubitWithZ)
    q[Zpi] = pulseShape
    q[Zpi2] = pulseShape/2
    q[Z3pi2] = -pulseShape/2
  end
  if q.pulseConvert[7] == -1
    q[I] = [0.0]
  end
  prepForSeq(q)
end


#============================ Basic seqeuncing ================================#
# The sequencing also has to be different depending on which type
# of waveform we are using.  In the case of FloatWaveforms, we need to insert
# idle pulses regularly, which are all the same point in the DAC's memory.
# In the case of ExactWaveforms, we need to access distinct points in the DAC's
# memory for each combination of (line, pulse).  The code allows
# for the possibility that the qubit's lines are not all on the same board.

# In preparing for 1-qubit sequencing, the qubit ensures that all pulses in its
# dictionary are up to date, and updates the 10x3 matrix of indices.
# These map the 10 basic single qubit pulses to their corresponding on-board
# waveform index.  That means that by calling q.pulseConvert[seq, n], with some
# sequence generated elsewhere in the code, we get the actual index list for 
# sequencing on the nth line.
function prepForSeq(q::Qubit)
  # Communicate with DAC only if necessary
  for p in q.waveforms
    if p[2].dirty
      q.pulseConvert[p[1],1] = pushWaveform(p[2].XYI, q.lineXYI[1], string(p[1])*"I")
      q.pulseConvert[p[1],2] = pushWaveform(p[2].XYQ, q.lineXYQ[1], string(p[1])*"Q")
      if isa(q, QubitWithZ)
        q.pulseConvert[p[1],3] = pushWaveform(p[2].Z, q.lineZ[1])
      end
      p[2].dirty = false
    end
  end
end

# Return the index of the waveform in memory
function pushWaveform(wavedata::Vector{UInt16}, board::Instrument, name="")
  if length(wavedata) == 0
    # This is one of the off channels of a FloatWaveform.  Just return the
    # stored index of idle pulse
    return idles[board]
  end
  if isa(board, InsAWG5014C)
    return awgPush(wavedata, board, name)
  else
    error("Instrument not supported yet, whine at a programmer")
  end
end

# We will be using the VISA protocol and binary writing.  We will also force
# every waveform to 250 points, with front-padded zeros.  A 30-point waveform
# will then have 220 points of offsetValue and then 30 points with signal.
function awgPush(wavedata::Vector{UInt16}, ins::InsAWG5014C, name::String)
  if length(wavedata) > 250
    error("Waveforms longer than 250 points not presently supported.")
  else
    binblockwrite(ins, "WLIST:WAV:DATA "*name*", "*reinterpret(UInt8,
      [fill(offsetValue, 250 - length(wavedata)); map(htol, wavedata)]))
  end
  name # Return the name, how the AWG will refer to it later.
end

# The following functions take pulse sequences in the form produced by the code,
# and send them to the DACs based on their indices in the DAC memory.  This
# means that sendSequence(q, [Xpi2, I, Xpi2]) for example, will send XYI, XYQ,
# and Z pulses according to the prior definitions of these pulses.  
function sendSequence(q::Qubit, sequence::Pulse)
  if size(sequence, 2) == 1
    sendSequence(q, sequence[:])
  else
    error("Please specify only one sequence\n")
  end
end


function sendSequence(q::Qubit, sequence::Vector{Int8})
  # Make sure we aren't going to send obsolete pulses
  # (no writes done if everything is current)
  prepForSeq(q)

  # TODO: More generalized version of this function.  For now, we assume the
  # qubit is controlled by the AWG, that it is a QubitNoZ object, and that
  # the other AWG lines need to idle until the sequence terminates, and then
  # perform a readout pulse.  This allows the sequencing to be done in one
  # (albeit large) command to the AWG.  If this isn't the goal, bug Brett.
  len = "SEQ:LENG "*string(length(sequence))*"\n"
  istring = map((x,y)-> "SEQ:ELEM"*string(x)*":WAV"*string(q.lineXYI[2])*" \""*y*"\"\n",
                                1:length(sequence), q.pulseConvert[sequence, 1])
  qstring = map((x,y)-> "SEQ:ELEM"*string(x)*":WAV"*string(q.lineXYQ[2])*" \""*y*"\"\n",
                                1:length(sequence), q.pulseConvert[sequence, 2])
  # Get the other channels, which are IQ for readout, and instruct them to be
  # idle at this time
  a = 1:4
  a = a[(a.!=q.lineXYI[2])&(a.!=q.lineXYQ[2])] # Select channels not used by XY
  idle1 = map(x -> "SEQ:ELEM"*string(x)*":WAV"*string(a[1])*" \"I\"\n")
  idle2 = map(x -> "SEQ:ELEM"*string(x)*":WAV"*string(a[1])*" \"I\"\n")
  # Put it all together and send to the AWG
  println(*(len, istring..., qstring..., idle1..., idle2...))
end
#=
function sendSequence(q::Qubit, sequence::Vector{Int8})
  # Make sure we aren't going to send obsolete pulses
  # (no writes done if everything is current)
  prepForSeq(q)
  # Call a lower level function for each line, using indices on the board
  # that correspond to the sequence requested
  seqLowLevel(q.pulseConvert[sequence, 1], q.lineXYI)
  seqLowLevel(q.pulseConvert[sequence, 2], q.lineXYQ)
  if isa(q, QubitWithZ)
    seqLowLevel(q.pulseConvert[sequence, 3], q.lineZ)
  end
  # Turn on output where along this path?

  # How to manage readout?  Do we send a trigger pulse along with the last
  # pulse in the seqeunce, or do we calculate the length of the pulse sequence
  # and just readout based on timing?  Also i expect not all sequences to be
  # sent are necessarily to be followed by readout, so I guess readout-
  # terminated sequences get their own method.
end
=#
# Nothing here covers readout or two-qubit gates.  Both are heirarchically
# lateral from the qubits themselves - pairs of coupled qubits need to know
# waveforms for 2-qubit gates, but not the qubits themselves.  The readout
# process covers all qubits connected to the readout line.  Whatever controls
# the readout will need to be able to tell which qubit is which, but the qubits
# themselves may remain unaware of the readout process.

#======================= Particle Swarm Optimization ==========================#
# For the optimization algorithm, I was wondering if particle-swarm stochastic
# algorithms would outperform those employed by other active groups.  The main
# advantage is that stochastic algorithms have a better expected performance on
# spaces with many parameters (I've seen PSO employed successfully on a 16-
# dimensional space with very narrow peaks in the fitness function).  With
# freedom to define many more parameters than previous groups, we could even
# try to find a meshing of the spectrum of the pulse and tune every point in
# the mesh independently.

# Explicitly, if we have N 14-bit digital points, converted to an analog signal
# with an appropriate low-pass filter, we could optimize all N points, or we
# could find the fourier components of the pulse and optimize them.  This has
# to be done subject to the constraints that the voltage is zero at T = 0 and
# T = pulselength, which I will assume is 20ns.  I will write waveformPSO to
# optimize the waveform point by point, subject to user-specified bounds.

# A PSO solution is a list of the values for all the parameters of interest.
# For us, this it is just be every point in the waveform.  Its fitness is
# determined by sending that pulse to the AWG as the definition of a gate, and
# benchmarking that gate using an ORBIT fitness routine.  For good SNR we
# need high averaging, so a single fitness measurement could take a couple
# milliseconds.  If we are waiting ~4 T1 every time, perhaps getting 5000 points
# could take on the order of a second.

function fitness(params::Vector{UInt16}, q::Qubit, p::Pulse)
  pulselength = div(length(params), 3)
  q[p] = ExactWaveform(params[1:pulselength], params[pulselength:2*pulselength],
                       params[2:pulselength:3*pulselength], true)
  pulseIndex = findfirst(SQPulse, p)

  # Every "1" in the readout detracts from the fitness score.
  1 - mean([sendSequenceAndReadout(q, benchmark1Qubit(
                                   pulseIndex, orbitSequenceLength))
        for _ in 1:orbitAveraging])
end

# Same but for FloatWaveforms, where we are optimizing vectors of Float64
function fitness(params::Vector{Float64}, q::Qubit, p::Pulse)
  q[p] = params
  pulseIndex = findfirst(SQPulse, p)
  1 - mean([sendSequenceAndReadout(q, benchmark1Qubit(
                                   pulseIndex, orbitSequenceLength))
        for _ in 1:orbitAveraging])
end

type Particle{T<:Vector}
  position::T
  currentFitness
  velocity::T
  bestPosition::T
  bestFitness
end

# Permit construction with no history
Particle{T<:Vector}(x::T, f, v::T) = Particle{T}(x, f, v, x, f)
# Permit construction with unknown fitness but specified qubit/pulse
Particle{T<:Vector}(x::T, v::T, q::Qubit, p::Pulse) =
                                             Particle{T}(x, fitness(x, q, p), v)

# Compare two particles.  Julia can find maximum element of an array so long as
# Base.isless is defined.
import Base.isless
Base.isless(x::Particle, y::Particle) = x.currentFitness < y.currentFitness

# Implement PSO where the parameters are given in a vector.

# By-point optimization of 20 UInt16s:
# popSize = 100
# selfWeight = neighborWeight = 1
# neighborhoodMin = 10
# inertiaMin = 0.007
# inertiaMax = 0.5
# we get convergence to the global optimum in ~75 iterations, equivalently
# 7500 fitness measurements.  Assuming 40 microseconds wait time for each
# readout, 5000 readouts per fitness measurement, and 0.1s latency for
# sending pulses to the machine, this comes out to about an hour.

# If we are using ExactWaveforms 20ns long, we need to optimize 60 UInt16s.
# We can run the same process as above with popsize tripled and expect to
# reach the global optimimum in about three times as long, totaling 9 hours.
# If it can be run overnight and get a by-point optimized gate I think that is
# worth it.

#
# With the same parameters but adjusting the fitness to seek out a 200-point
# function rather than a 20-point function, we converge in about 350 iterations
#
# The pop size should be larger than the number of arguments in all cases
# The inertia, selfWeight and neighborWeight should be of comparable magnitude
# TODO: make argument list less overwhelming?
# Ideally, user specifies just the qubit and pulse, the desired pulse length
# and whether they want to assume inactive lines idle.  They possibly also
# input a seed waveform with known decent performance (speeds up the algorithm).
# Then the program determines good parameters, bases the bounds purely on the
# bounds of the DAC, and churns away.  It should also store locally the best
# found gate so far, and perhaps recompute its fitness sporadically to make sure
# it is signal and not just noise.


function waveformPSO{T<:Real}(
          q::Qubit, p::Pulse,
          popSize::Integer,
          boundsMin::Vector{T}, boundsMax::Vector{T},
          neighborhoodMin::Integer,
          maxIterations::Integer,
          inertiaMin=0.007, inertiaMax=0.5,
          selfWeight=1, neighborWeight=1)

  # Initialize a pool of "Popsize" Particles, with each element in the bounds
  pinit = hcat(map((x,y) -> x + (y-x)*rand(popSize), boundsMin, boundsMax)...)'
  vinit = hcat(map(x -> 2x - x*rand(popSize), boundsMax - boundsMin)...)'
  populationInfo = [Particle(pinit[:,i], vinit[:,i], q, p)
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
    map((x,y) -> x.velocity = W*x.velocity +
           selfWeight*rand(length(x.position)).*(x.bestPosition - x.position)
         + neighborWeight*rand(length(x.position)).*(y.position - x.position)
         , populationInfo, localWinners)
  
    # Update the positions based on these new velocities (can you tell I prefer
    # functional programming?). We clip it to the proper range.
    map(x -> x.position = map((p,lo,hi) -> (lo + hi + abs(p-lo) - abs(p-hi))/2,
        x.position + x.velocity, boundsMin, boundsMax), populationInfo)
  
    # Update the current fitness.  If it is better than the old fitness, save
    # current position.  If it is the best seen so far, udpate best fitness.
    # Done sequentially since the fitness calls can't be parallelized.
    for x in populationInfo
        x.currentFitness = fitness(x.position, q, p)
        if (x.currentFitness > x.bestFitness)
          x.bestPosition = x.position
          x.bestFitness = x.currentFitness
        end
        if (x.currentFitness > winnerF)
          winnerX, winnerF = x.position, x.currentFitness
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
  winnerX, winnerF
end

#============================== AWG specifics =================================#
# So the AWG is kind of sucky, and requires all waveforms be 250 points or more.
# There are two ways to work around this.  We can have one waveform per pulse,
# just accepting that the pulses are about 10 times too long.  Or we can try to
# have every sequence be a different waveform.  That means uploading a lot of
# data to the AWG, and might be a considerable slowdown.  One can define up to
# 32K distinct waveforms, so a third option might be to put enough pulses in
# one waveform to have 32K distinct shapes - with 1 qubit and 10 basic pulses
# all possible sets of 4 pulses can fit easily.  It would be a real pain though
# to update one pulse, since it appears in so many different waveforms.

# For ease of ORBIT and similar tactics, I think the first way is the most
# practical - it minimizes the amount of writes to the AWG.  For later use with
# algorithms, the third approach might instead be the best.  Hopefully though
# we are away from the AWG by that point.  FPGAs would be far cheaper for many-
# qubit systems.



function awgSeq()

end

# The reality of working with the AWG - we are limited to two lines of control.
# The readout pulse requires phase sensitivity and thus IQ mixing, using two of
# the four lines.  That means that we have to perform the single-qubit 
# clifford gates using no Z control.  Until we have FPGA control, all our qubits
# will be QubitNoZ type.

# I would like to see how quickly a new sequence can be loaded to the AWG.
# Ideally, it requires very few writes and can be done in the tens of
# microseconds when the qubit is T1-relaxing to the ground state.  Realistically
# though, the AWG is not made for this.  The communication protocol relies on
# sending waveforms by name and sequencing one index at a time.  To send a 30-
# gate sequence would require a command string such as the following
# "SEQ:LENG 31"
# "SEQ:ELEM1:WAV1 "Xpi2I""
# "SEQ:ELEM1:WAV2 "Xpi2Q""
# "SEQ:ELEM2:WAV1 "YpiI""
# "SEQ:ELEM2:WAV2 "YpiQ""
# ...
# "SEQ:ELEM30:WAV1 "XpiI""
# "SEQ:ELEM30:WAV2 "XpiQ""
# which is 61 individual commands.

# The commands may be sent as a single string, which could be nicer for latency.
# The waveforms need to have names (instead of indices), so I guess when a qubit
# has the AWG as its instrument, the 7x2 matrix actually stores strings.