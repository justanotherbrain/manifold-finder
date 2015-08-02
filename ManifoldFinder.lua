require 'image'
require 'nn'
require 'inn'
require 'cunn'
require 'cudnn'
if (pcall(require,'cutorch-rtc')) then dofile 'roll.lua' end

torch.setdefaulttensortype('torch.FloatTensor')

perchannel_mean = {103.939, 116.779, 128.68}
Normalization = {mean=118.380948/255, std = 61.896913/255}

-- Load VGG-19
model_path = '/home/mad573/research/vis/deep-dreams/model/'
proto = model_path.. 'VGG_ILSVRC_19_layers_deploy.prototxt'
caffemodel = model_path..'VGG_ILSVRC_19_layers.caffemodel'

if io.open('ilsvrc_19.net',"r") then
    net = torch.load('ilsvrc_19.net')
else
    print('If you are on HPC, load torch/intel/20150218 - you will run into an error after... but at least you will have the network loaded and you can start over')
    require 'loadcaffe'
    net = loadcaffe.load(proto, caffemodel, pcall(require,'cudnn') and 'cudnn' or 'nn'):cuda()
    torch.save('ilsvrc_19.net',net)
end

print(tostring(net))

local m = nn.Sequential()
for i,v in ipairs(net.modules) do
    if i == 37 then
        m:add(nn.SpatialAdaptiveMaxPooling(7,7):cuda())
    else
        m:add(v)
    end
end
net = m

function GetGoalStats(net, guide, end_layer, clip, step_size, jitter)
    local step_size = step_size or 1.5
    local jitter = jitter or 32
    local layer = layer or 15
    local clip = clip
    if clip == nil then clip = true end

    local ox = 2*jitter - math.random(jitter)
    local oy = 2*jitter - math.random(jitter)
    
    local goal = net:forward(guide)
    return goal
end


function make_step(net, img, goal, end_layer, clip, step_size, jitter)
    local step_size = step_size or 1.5
    local jitter = jitter or 32
    local end_layer = end_layer or 15
    local clip = clip
    if clip == nil then clip = true end

    local ox = 2*jitter - math.random(jitter)
    local oy = 2*jitter - math.random(jitter)

    if roll then roll(img, img:clone(), ox, oy) end
    --local dst = net:forward(img)
    
    -- don't need to call backward
    --local g = net:updateGradInput(img,dst)
    g = net:updateGradInput(img,goal)
    -- apply normalized ascent step to input image
    img:add(g:mul(step_size/torch.abs(g):mean()))

    if roll then roll(img, img:clone(), -ox, -oy) end
    if clip then
        bias = torch.Tensor(perchannel_mean):mean()
        img:clamp(-bias, 255-bias)
    end
    return img
end

function preprocess(base_img)
    local im = base_img:clone()
    im[1]:copy(base_img[3])
    im[3]:copy(base_img[1])
    im:mul(255)
    for i=1,3 do im[i]:add(-perchannel_mean[i]) end
    return im
end

function undopreprocess(src)
    src = src:clone()
    for i=1,3 do src:select(1,i):add(perchannel_mean[i]) end
    local src2 = src:clone()
    src2[1]:copy(src[3])
    src2[3]:copy(src[1])
    return src2:div(255)
end

function ImageSynthesis(net, guide, init, iter_n, octave_n, octave_scale, end_layer, clip, visualize)
    
    local function reduceNet(full_net, end_layer)
        local net = nn.Sequential()
        for l=1,end_layer do
            net:add(full_net:get(l))
        end
        return net
    end

    local iter_n = iter_n or 10
    local octave_n = octave_n or 4
    local octave_scale = octave_scale or 1.4
    local end_layer = end_layer or 20
    local net = reduceNet(net, end_layer)
    local clip = clip
    local init = init
    if clip == nil then clip = true end
    
    
    -- prepare base image for octaves
    local octaves = {}

    local _,h,w  = unpack(guide:size():totable())
    
    if init == nil then 
        init = torch.rand(3,h,w) 
    elseif init == 'black' then
        init = torch.Tensor(3,h,w):zero()
    elseif init == 'white' then
        init = torch.Tensor(3,h,w):fill(255)
    end

    octaves[octave_n] = preprocess(init)

    for i=octave_n-1,1,-1 do
        octaves[i] = image.scale(octaves[i+1], math.ceil((1/octave_scale)*w), math.ceil((1/octave_scale)*h),'bilinear')
    end

    local detail
    local src
    
    local goal = GetGoalStats(net, guide:cuda(), end_layer, clip):float()
    
    for octave, octave_base in pairs(octaves) do
        src = octave_base
        local _,h1,w1 = unpack(src:size():totable())
        if octave > 1 then
            -- upscale details from previous octave
            src:add(image.scale(detail, w1, h1, 'simple'))
            --guide:add(image.scale())
        end
        for i=1,iter_n do
            src = make_step(net, src:cuda(), goal:cuda(), end_layer, clip):float()
        end
        -- extract details produced on current octave
        detail = src-octave_base
    end
    collectgarbage()
    src = undopreprocess(src)
    return src
end


function demo(n_iter)
    --img = torch.Tensor(3,360,360):zero()
    guide = image.load('guide.png')
    --x = ImageSynthesis(net, guide, nil, 1, 1, 1, 1)
    x = ImageSynthesis(net, guide, 'black', 1,1,1,1)
    for i=1,n_iter do
        x = ImageSynthesis(net,guide, x, 1,1,1,1)
    end
    image.save('test.png',x)
end

--demo()



