module ORBIT

using ..ParticleSwarm
using ..Sequencing # Includes Clifford, Waveforms, and Qubit

# Fitness calls here are of the form
# "How well does <waveform> implement <pulse> on <qubit>?"

# The overarching "ORBIT" call will include a qubit and a gate, and define the
# fitness anonymously through the below.
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


function optimize(q::Qubit, p::Pulse, numpoints::Integer, init::Waveform)
  if isa(init, ExactWaveform)
    if isa(q, QubitWithZ)
      varCount = 3*numpoints
      guess = [init.XYI; init.XYQ; init.Z]
    else
      varCount = 2*numpoints
      guess = [init.XYI; init.XYQ]
    end
  else
    varCount = numpoints
    guess = init.wavedata
  end

  if p[1] < 7 && isa(q.lineXYI[1], InstrumentControl.InsAWG5014C) # XY on AWG
end

end # End module ORBIT