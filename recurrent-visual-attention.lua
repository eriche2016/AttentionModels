require "dp"
require "rnn"

-- References :
-- A. https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua
-- B. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- C. http://incompleteideas.net/sutton/williams-92.pdf

-- Disclaimer: this is basically the same code as found in A., but with notes added
-- about what is happening at each step (so that I could understand the model better)


--[[Command Line Arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text("Train a Recurrent Model for Visual Attention")
cmd:text("Example:")
cmd:text("$> th recurrent-visual-attention.lua > results.txt")
cmd:text("Options:")

--[[General Options]]--
cmd:option("--learningRate", 0.01, "learning rate at t=0")
cmd:option("--minLR", 0.00001, "minimum learning rate")
cmd:option("--saturateEpoch", 800, "epoch at which linear decayed LR will reach minLR")
cmd:option("--momentum", 0.9, "momentum")
cmd:option("--maxOutNorm", -1, "max norm each layers output neuron weights")
cmd:option("--cutoffNorm", -1, "max l2-norm of contatenation of all gradParam tensors")
cmd:option("--batchSize", 20, "number of examples per batch")
cmd:option("--cuda", false, "use CUDA")
cmd:option("--useDevice", 1, "sets the device (GPU) to use")
cmd:option("--maxEpoch", 2000, "maximum number of epochs to run")
cmd:option("--maxTries", 100, "maximum number of epochs to try to find a better local minima for early-stopping")
cmd:option("--transfer", "ReLU", "activation function")
cmd:option("--uniform", 0.1, "initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization")
cmd:option("--xpPath", "", "path to a previously saved model")
cmd:option("--progress", false, "print progress bar")
cmd:option("--silent", false, "dont print anything to stdout")

--[[Glimpse Options]]--
cmd:option("--glimpseHiddenSize", 128, "size of glimpse hidden layer")
cmd:option("--glimpsePatchSize", 8, "size of glimpse patch at highest res (height = width)")
cmd:option("--glimpseScale", 2, "scale of successive patches w.r.t. original input image")
cmd:option("--glimpseDepth", 1, "number of concatenated downscaled patches")
cmd:option("--locatorHiddenSize", 128, "size of locator hidden layer")
cmd:option("--imageHiddenSize", 256, "size of hidden layer combining glimpse and locator hiddens")

--[[Recurrent Options]]--
cmd:option("--hiddenSize", 256, "number of hidden units used in Simple RNN.")
cmd:option("--FastLSTM", false, "use LSTM instead of linear layer")
cmd:option("--rho", 7, "back-propagate through time (BPTT) for rho time-steps")

--[[Data Options]]--
cmd:option("--dataset", "Mnist", "which dataset to use: Mnist | TranslatedMnist | ...")
cmd:text()

--[[REINFORCE Options]]--
cmd:option("--locatorStd", 0.11, "stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)")
cmd:option("--stochastic", false, "Reinforce modules forward inputs stochastically during evaluation")
cmd:option("--rewardScale", 1, "scale of positive reward (negative is 0)")
cmd:option("--unitPixels", 13, "the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)")

local opt = cmd:parse(arg or {})
if not opt.silent then
	table.print(opt)
end

--[[Load Data]]--
if opt.dataset == "TranslatedMnist" then
	ds = torch.checkpoint(
		paths.concat(dp.DATA_DIR, "checkpoint/dp.TranslatedMnist.t7"),
		function() return dp[opt.dataset]() end,
		opt.overwrite
	)
else
	ds = dp[opt.dataset]()
end

--[[Saved model]]--
if opt.xpPath ~= "" then
	-- Look for saved model
	assert(paths.filep(opt.xpPath), opt.xpPath.." does not exist")
	-- Load CUDA and Optim if using GPU
	if opt.cuda then
		require "optim"
		require "cudann"
		cutorch.setDevice(opt.useDevice)
	end
	-- Load model
	xp = torch.load(cmd.xpPath)
	if opt.cuda then
		xp:cuda()
	else
		xp:float()
	end
	-- Run model on dataset
	print"running"
	xp:run(ds)
	os.exit()
end

--[[---------]]--
--[[--Model--]]--
--[[---------]]--

--[[Location Sensor: maps given location to hidden space]]--
locationSensor = nn.Sequential()
-- This layer selects the second item (i.e., the location) from the input table
locationSensor:add(nn.SelectTable(2))
-- This layer applies a linear transformation 
-- and goes from the input coordinates to opt.locatorHiddenSize hidden units
locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
-- This layer applies the activation function
locationSensor:add(nn[opt.transfer]())

--[[Glimpse Sensor: Produces "retina-like" representation, given image+location]]--
glimpseSensor = nn.Sequential()
-- DontCast prevents this layer from being cast when type() is called
-- SpatialGlimpse can be used to focus attention using {image, location}
glimpseSensor:add(nn.DontCast(nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale):float(), true))
-- Collapse all non-batch dimensions in preparation for linear transform
glimpseSensor:add(nn.Collapse(3))
-- imageSize("c") gets the size of the input along color axis?
-- Linear transformation from an input layer with imageSize(c)*glimpsePatchSize^2*glipseDepth to hidden layer of size glipseHiddenSize
glimpseSensor:add(nn.Linear(ds:imageSize("c")*(opt.glimpsePatchSize^2)*opt.glimpseDepth, opt.glimpseHiddenSize))
glimpseSensor:add(nn[opt.transfer]())

--[[Glimpse Network: Combines information from the glimpse sensor and location sensor]]--
--[[Produces glimpse feature vector g[t] ]]--
glimpse = nn.Sequential()
-- This will apply each input to both locationSensor and glimpseSensor
glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
-- Joins the result of processing the inputs with locationSensor and glimpseSensor
glimpse:add(nn.JoinTable(1,1))
-- Linear transformation of the result
glimpse:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.imageHiddenSize))
glimpse:add(nn[opt.transfer]())
-- Emit g[t]
glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))

--[[Core Network]]--
--[[ Keeps track of the current hidden state h[t], conditioned on h[t-1] and g[t] ]]--
if opt.FastLSTM then
  recurrent = nn.FastLSTM(opt.hiddenSize, opt.hiddenSize)
else
  recurrent = nn.Linear(opt.hiddenSize, opt.hiddenSize)
end
rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn[opt.transfer](), 99999)

--[[Check that images are square]]--
imageSize = ds:imageSize("h")
assert(ds:imageSize("h") == ds:imageSize("w"))

--[[Location network]]--
--[[Samples the next location given previous hidden state]]--
--[[Trained using REINFORCE algorithm]]--
locator = nn.Sequential()
-- Linearly transforms the hidden state to a location
locator:add(nn.Linear(opt.hiddenSize, 2))
-- bounds mean between -1 and 1
locator:add(nn.HardTanh()) 
-- sample from normal, uses REINFORCE learning rule
locator:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) 
assert(locator:get(3).stochastic == opt.stochastic, "Please update the dpnn package : luarocks install dpnn")
-- bounds sample between -1 and 1
locator:add(nn.HardTanh()) 
locator:add(nn.MulConstant(opt.unitPixels*2/ds:imageSize("h")))

--[[Packages together the location and core network]]--
attention = nn.RecurrentAttention(rnn, locator, opt.rho, {opt.hiddenSize})

--[[Agent (Action network)]]--
--[[Outputs {classpred, {classpred, basereward}}]]--
agent = nn.Sequential()
-- Convert dataset to batch x color x height x width format
agent:add(nn.Convert(ds:ioShapes(), "bchw"))
-- Add location + core network to agent
agent:add(attention)
-- Classifier used to determine labels based on h[t]
agent:add(nn.SelectTable(-1))
agent:add(nn.Linear(opt.hiddenSize, #ds:classes()))
agent:add(nn.LogSoftMax())
-- Add baseline reward predictor to agent
seq = nn.Sequential()
seq:add(nn.Constant(1,1))
seq:add(nn.Add(1))
concat = nn.ConcatTable():add(nn.Identity()):add(seq)
concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)
agent:add(concat2)

-- Initialize Parameters --
if opt.uniform > 0 then
   for k,param in ipairs(agent:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

--[[Propagators]]--
-- Initialize decay factor
opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch

-- 
train = dp.Optimizer{
   loss = nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
      :add(nn.ModuleCriterion(nn.VRClassReward(agent, opt.rewardScale), nil, nn.Convert())) -- REINFORCE
   ,
   epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         opt.learningRate = opt.learningRate + opt.decayFactor
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
         end
      end
   end,
   callback = function(model, report)
      if opt.cutoffNorm > 0 then
         local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
         if opt.lastEpoch < report.epoch and not opt.silent then
            print("mean gradParam norm", opt.meanNorm)
         end
      end
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams
   end,
   feedback = dp.Confusion{output_module=nn.SelectTable(1)},
   sampler = dp.ShuffleSampler{
      epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
   },
   progress = opt.progress
}


valid = dp.Evaluator{
   feedback = dp.Confusion{output_module=nn.SelectTable(1)},
   sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = opt.batchSize},
   progress = opt.progress
}
if not opt.noTest then
   tester = dp.Evaluator{
      feedback = dp.Confusion{output_module=nn.SelectTable(1)},
      sampler = dp.Sampler{batch_size = opt.batchSize}
   }
end

--[[Experiment]]--
xp = dp.Experiment{
   model = agent,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = {
      ad,
      dp.FileLogger(),
      dp.EarlyStopper{
         max_epochs = opt.maxTries,
         error_report={"validator","feedback","confusion","accuracy"},
         maximize = true
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
   require "cutorch"
   require "cunn"
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Agent :"
   print(agent)
end

xp.opt = opt

xp:run(ds)