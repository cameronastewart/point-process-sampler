import sys
import argparse
import time
import numpy as np
from collections import deque

# config
BURN_IN = 1000000
BATCH_SIZE = 3000

# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("sampler", help="choose the sampler", choices=["point", "birthdeath", "sqrt", "metropolis", "barker"])
parser.add_argument("model", help="choose the model to sample from", choices=["poisson", "ising", "neural"])
parser.add_argument("scale", help="adjust the scaling parameter (lambda, beta, or alpha)", type=float)
parser.add_argument("run", help="set the run number (this determines the random seed)", type=int)
args = parser.parse_args()

# set random seed (first five digits of pi + args.run)
np.random.seed(31415 + args.run)

# load and set parameters
if args.model == "poisson":
    dim = 1
    parameters = {"lambda": np.array([args.scale])}
elif args.model == "ising":
    dim = 100
    try:
        parametersNpz = np.load("ising_parameters.npz")
    except OSError:
        print("Error: cannot open ising_parameters.npz")
        sys.exit(1)
    parameters = {"weights": args.scale * parametersNpz["weights"],
                  "biases": args.scale * parametersNpz["biases"]}
elif args.model == "neural":
    dim = 100
    try:
        parametersNpz = np.load("neural_parameters.npz")
    except OSError:
        print("Error: cannot open neural_parameters.npz")
        sys.exit(1)
    parameters = {"weights": args.scale * parametersNpz["weights"],
                  "biases": parametersNpz["biases"],
                  "a1": 1,
                  "a0": 0}

# point process sampler
def pointProcess(dim, model, parameters):
    # variable "state" is the queue length of the point process sampler
    simTime = 0
    state = np.zeros(dim)
    pointHistory = deque()
    states = []
    stateWeightsUnnormalised = []

    for i in range(BURN_IN + BATCH_SIZE ** 2):
        if i == BURN_IN:
            startTime = time.process_time()
        if i >= BURN_IN:
            states.append(state.copy())
            oldTime = simTime
        
        # calculate conditional intensities
        if model == "poisson":
            rate = parameters["lambda"]
        elif model == "ising":
            rate = (1 - state) * np.exp(2 * parameters["weights"] @ state - parameters["biases"])
        elif model == "neural":
            rate = np.exp(parameters["weights"] @ state + parameters["biases"] - np.exp(parameters["a1"] * state + parameters["a0"]))
        
        # propose a new point, check for points leaving the sliding window, then update the state
        rateSum = rate.sum()

        with np.errstate(divide="ignore"):
            nextPoint = np.random.exponential(np.float64(1) / rateSum)

        if len(pointHistory) == 0 or simTime + nextPoint < pointHistory[0]["simTime"] + 1:
            simTime += nextPoint
            component = np.random.choice(dim, p=rate / rateSum)
            pointHistory.append({"component": component, "simTime": simTime})
            state[component] += 1
        else:
            removedPoint = pointHistory.popleft()
            simTime = removedPoint["simTime"] + 1
            state[removedPoint["component"]] -= 1
        
        # states are weighted by holding times
        if i >= BURN_IN:
            stateWeightsUnnormalised.append(simTime - oldTime)

    elapsedTime = time.process_time() - startTime

    return stateWeightsUnnormalised, states, elapsedTime

# birth-death and Zanella samplers
def ctmc(dim, sampler, model, parameters):
    simTime = 0
    state = np.zeros(dim)
    states = []
    stateWeightsUnnormalised = []

    for i in range(BURN_IN + BATCH_SIZE ** 2):
        if i == BURN_IN:
            startTime = time.process_time()
        if i >= BURN_IN:
            states.append(state.copy())
            oldTime = simTime
        
        # calculate transition rates
        if sampler == "birthdeath":
            if model == "poisson":
                arrivalRate = parameters["lambda"]
                departureRate = state
            elif model == "ising":
                arrivalRate = (1 - state) * np.exp(2 * parameters["weights"] @ state - parameters["biases"])
                departureRate = state
            elif model == "neural":
                arrivalRate = np.exp(parameters["weights"] @ state + parameters["biases"] - np.exp(parameters["a1"] * state + parameters["a0"]))
                departureRate = state
        else:
            if model == "poisson":
                zArrival = parameters["lambda"] / (state + 1)
                zDeparture = state / parameters["lambda"]
            elif model == "ising":
                rateIntermediate = 2 * parameters["weights"] @ state - parameters["biases"]
                zArrival = (1 - state) * np.exp(rateIntermediate)
                zDeparture = state * np.exp(-rateIntermediate)
            elif model == "neural":
                rateIntermediate = parameters["weights"] @ state + parameters["biases"]
                zArrival = np.exp(rateIntermediate - np.exp(parameters["a1"] * state + parameters["a0"])) / (state + 1)
                zDeparture = state * np.exp(-rateIntermediate + parameters["weights"].diagonal() + np.exp(parameters["a1"] * (state - 1) + parameters["a0"]))

            # apply Zanella balancing functions
            if sampler == "sqrt":
                arrivalRate = zArrival ** 0.5
                departureRate = zDeparture ** 0.5
            elif sampler == "metropolis":
                arrivalRate = np.minimum(1, zArrival)
                departureRate = np.minimum(1, zDeparture)
            elif sampler == "barker":
                arrivalRate = zArrival / (1 + zArrival)
                departureRate = zDeparture / (1 + zDeparture)

        # propose arrival and departure times, accept whichever happens first
        arrivalRateSum = arrivalRate.sum()
        departureRateSum = departureRate.sum()

        with np.errstate(divide="ignore"):
            nextArrival = np.random.exponential(np.float64(1) / arrivalRateSum)
            nextDeparture = np.random.exponential(np.float64(1) / departureRateSum)

        if nextArrival < nextDeparture:
            simTime += nextArrival
            arrivalComponent = np.random.choice(dim, p=arrivalRate / arrivalRateSum)
            state[arrivalComponent] += 1
        else:
            simTime += nextDeparture
            departureComponent = np.random.choice(dim, p=departureRate / departureRateSum)
            state[departureComponent] -= 1
        
        # states are weighted by holding times
        if i >= BURN_IN:
            stateWeightsUnnormalised.append(simTime - oldTime)

    elapsedTime = time.process_time() - startTime

    return stateWeightsUnnormalised, states, elapsedTime

# run samplers and return process trajectories (after burn-in)
if args.sampler == "point":
    stateWeightsUnnormalised, states, elapsedTime = pointProcess(dim, args.model, parameters)
else:
    stateWeightsUnnormalised, states, elapsedTime = ctmc(dim, args.sampler, args.model, parameters)

# calculates mutlivariate ESS
def ess(stateWeightsUnnormalised, states, dim):
    stateWeightsUnnormalisedArray = np.array(stateWeightsUnnormalised)
    statesArray = np.array(states)

    # calculate batch means
    stateWeightsUnnormalisedBatched = np.tile(stateWeightsUnnormalisedArray.reshape((BATCH_SIZE, BATCH_SIZE, 1)), (1, 1, dim))
    statesBatched = statesArray.reshape((BATCH_SIZE, BATCH_SIZE, dim))
    batchMeans = np.average(statesBatched, axis=1, weights=stateWeightsUnnormalisedBatched)

    # calculate covariance and asymptotic covariance matrices, return ESS
    xiCov = np.cov(statesArray, rowvar=False, aweights=stateWeightsUnnormalisedArray).reshape((dim, dim))
    sigmaCov = BATCH_SIZE * np.cov(batchMeans, rowvar=False).reshape((dim, dim))
    _, xiLogDet = np.linalg.slogdet(xiCov)
    _, sigmaLogDet = np.linalg.slogdet(sigmaCov)

    return BATCH_SIZE ** 2 * np.exp((xiLogDet - sigmaLogDet) / dim)

# calculate and append output to file
essEstimate = ess(stateWeightsUnnormalised, states, dim)

file = open(f"{args.sampler}_{args.model}_{args.scale}.csv", "a")
file.write(str(essEstimate) + "," + str(elapsedTime) + "\n")
file.close()