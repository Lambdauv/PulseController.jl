module ORBIT

using ..ParticleSwarm
using ..Sequencing # Includes Clifford, Waveforms, and Qubit
using ..Waveforms

export tuneGate


# Fitness calls here are of the form
# "How well does <waveform> implement <pulse> on <qubit>?"

# The overarching "ORBIT" call will include a qubit and a gate, and define the
# fitness anonymously through the below.
#function fitness(params::Vector{UInt16}, q::Qubit, p::Pulse)
#  pulselength = div(length(params), 3)
#  q[p] = ExactWaveform(params[1:pulselength], params[pulselength:2*pulselength],
#                       params[2:pulselength:3*pulselength], true)
#  pulseIndex = findfirst(SQPulse, p)
#
#  # Every "1" in the readout detracts from the fitness score.
#  1 - mean([sendSequenceAndReadout(q, benchmark1Qubit(
#                                   pulseIndex, orbitSequenceLength))
#        for _ in 1:orbitAveraging])
#end

# Same but for FloatWaveforms, where we are optimizing vectors of Float64
#function fitness(params::Vector{Float64}, q::Qubit, p::Pulse)
#  q[p] = params
#  pulseIndex = findfirst(SQPulse, p)
#  1 - mean([sendSequenceAndReadout(q, benchmark1Qubit(
#                                   pulseIndex, orbitSequenceLength))
#        for _ in 1:orbitAveraging])
#end

const numClifs = 80
const numSeq = 6
import InstrumentControl: IQSoftwareResponse, Stimulus, Response, Instrument, source, measure, sweep, status, Done
type rbStim <: Stimulus
    q::Qubit
    r::Readout
    p::Pulse
    ind::Integer
end

import ..Clifford: SQPulseNoZ, Ypi
function rbStim(q::Qubit, r::Readout, p::Pulse)
    ind = findfirst(SQPulseNoZ, p)
    rbStim(q, r, p, ind)
end

function source(x::rbStim, g::ExactWaveform)
    gprime = ExactWaveform(g.XYQ, 16382-g.XYI, UInt16[], false, true)
    x.q[x.p] = g
    x.q[Ypi] = gprime
    sendNormalSequence(x.q, benchmark1Qubit(numClifs, x.ind, false), x.r)
end

#const binary1 = 28.26*cis(-0.786)
#const binary0 = 27.62*cis(-0.513)
# 1 microsecond readout has different numbers
const binary0 = 49.4 - 26.1im
const binary1 = 41.3 - 40.2im
binaryOut(IQdata) = abs(IQdata-binary1) < abs(IQdata-binary0) ? 1 : 0
type zeroProb <: Response
    r::IQSoftwareResponse
end

function measure(x::zeroProb)
    m = measure(x.r)::Array{Complex{Float32},1}
    1-mean(binaryOut.(m))
end

function fitnessMap(candidates::Vector{ExactWaveform}, stim::rbStim, out::zeroProb)
    pool = length(candidates)
    theJob = sweep(out, (stim, vcat(fill(candidates, numSeq)...)))
    while status(theJob) != Done
        sleep(1)
    end
    bare = theJob.sweep.result.data
    return [mean(bare[a:pool:(numSeq*pool)]) for a in 1:pool]
end


toParams(x::ExactWaveform) = Int64[x.XYI; x.XYQ]
function fromParams(x::Vector{Int64})
    numpoints = div(length(x), 2)
    y = UInt16.(x)
    ExactWaveform(y[1:numpoints], y[(numpoints+1):(2*numpoints)], UInt16[], false, true)
end

# Generalize later.  For now, let's just get it working with AWG and
# a lot of assumptions.
function tuneGate(q::Qubit, p::Pulse, r::Readout, iq::IQSoftwareResponse)
  init = q[p]
  numpoints = length(init.XYI)

  rb = rbStim(q, r, p)
  zp = zeroProb(iq)
  fitness(pool) = fitnessMap(fromParams.(pool), rb, zp)
  lowBound = map(x -> max(x - 800, 0), toParams(init))
  hiBound = map(x -> min(x + 800, 16382), toParams(init))
  winner, score = PSO(fitness, lowBound, hiBound, candidate=toParams(init))
  q[p] = fromParams(winner)
  return score

end

end # End module ORBIT
