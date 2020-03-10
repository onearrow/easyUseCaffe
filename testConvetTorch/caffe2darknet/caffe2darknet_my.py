#!/home/ubuntu/anaconda2/bin/python -f 

from collections import OrderedDict
from cfg import *
from prototxt import *
import numpy as np

def merge_active(layers):
    newlayers = []
    i=0
    layer_num = len(layers)
    while i < layer_num:
        layer = layers[i]
        print(i, layer['name'], layer['type'])
        if layer['type'] == 'Convolution' or layer['type'] == 'DepthwiseConvolution':
            if i + 2 < layer_num and layers[i + 1]['type'] == 'BatchNorm' and layers[i + 2]['type'] == 'Scale':
                print(i + 1, layers[i + 1]['name'], layers[i + 1]['type'])
                print(i + 2, layers[i + 2]['name'], layers[i + 2]['type'])
                layer['batch_normalize'] = '1'
                bn_layer = layers[i + 1]
                scale_layer = layers[i + 2]
                last_layer = scale_layer
                i = i + 2
            else:
                layer['batch_normalize'] = '0'

            if i + 1 < layer_num and layers[i + 1]['type'] == 'ReLU':
                print(i + 1, layers[i + 1]['name'], layers[i + 1]['type'])
                act_layer = layers[i + 1]
                layer['activation'] = 'relu'
                top = act_layer['top']
                newlayers.append(layer)
                i = i + 1
            elif i + 1 < layer_num and layers[i + 1]['type'] == 'PReLU':
                print(i + 1, layers[i + 1]['name'], layers[i + 1]['type'])
                act_layer = layers[i + 1]
                layer['activation'] = 'Prelu'
                top = act_layer['top']
                newlayers.append(layer)
                i = i + 1
            else:
                layer['activation'] = 'linear'
                top = last_layer['top']
                newlayers.append(layer)
            i = i + 1
        else:
            newlayers.append(layer)
            
#        if layer['type'] == 'DepthwiseConvolution':
#            if i + 2 < layer_num and layers[i + 1]['type'] == 'BatchNorm' and layers[i + 2]['type'] == 'Scale':
#                print(i + 1, layers[i + 1]['name'], layers[i + 1]['type'])
#                print(i + 2, layers[i + 2]['name'], layers[i + 2]['type']) 
#                layer['batch_normalize'] = '1'
#                bn_layer = layers[i + 1]
#                scale_layer = layers[i + 2]
#                last_layer = scale_layer
#                i = i + 2
#            else:
#                layer['batch_normalize'] = '0'
#                
#            if i + 1 < layer_num and layers[i+1]['type'] == 'ReLU':
#                print(i + 1, layers[i + 1]['name'], layers[i + 1]['type'])
#                act_layer = layers[i + 1]
#                layer['activation'] = 'Prelu'
#                top = act_layer['top']
#                newlayers.append(layer)
#                i = i + 1
#            else:
#                layer['activation'] = 'linear'
#                top = last_layer['top']
#                newlayers.append(layer)
#            i = i + 1
#        else:
#            newlayers.append(layer)
#                
            

def caffe2darknet(protofile, caffemodel):
    model = parse_caffemodel(caffemodel)
    layers = model.layer
    if len(layers) == 0:
        print('Using V1LayerParameter')
        layers = model.layers
    
    lmap = {}
    for l in layers:
        lmap[l.name] = l

    net_info = parse_prototxt(protofile)
    props = net_info['props']

    wdata = []
    blocks = []
    block = OrderedDict()
    block['type'] = 'net'
#    if props.has_key('input_shape'):
    if 'input_shape' in props:
        block['batch'] = props['input_shape']['dim'][0]
        block['channels'] = props['input_shape']['dim'][1]
        block['height'] = props['input_shape']['dim'][2]
        block['width'] = props['input_shape']['dim'][3]
    else:
        block['batch'] = props['input_dim'][0]
        block['channels'] = props['input_dim'][1]
        block['height'] = props['input_dim'][2]
        block['width'] = props['input_dim'][3]
#    if props.has_key('mean_file'):
    if 'mean_file' in props:
        block['mean_file'] = props['mean_file']
    blocks.append(block)

    layers = net_info['layers']
    layer_num = len(layers)
    i = 0 # layer id
    layer_id = dict()
    layer_id[props['input']] = 0
    while i < layer_num:
        layer = layers[i]
        print(i,layer['name'], layer['type'])
        if layer['type'] == 'Convolution':
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                print("route", layer_id[layer['bottom']], len(blocks), block['layers'])
                blocks.append(block)
            #assert(i+1 < layer_num and layers[i+1]['type'] == 'BatchNorm')
            #assert(i+2 < layer_num and layers[i+2]['type'] == 'Scale')
            conv_layer = layers[i]
            block = OrderedDict()
            block['type'] = 'convolutional'
            block['filters'] = conv_layer['convolution_param']['num_output']
            block['size'] = conv_layer['convolution_param']['kernel_size']
            if 'stride' in conv_layer['convolution_param']:
                block['stride'] = conv_layer['convolution_param']['stride']
            else:
                block['stride'] = '1'  # conv_layer['convolution_param']['stride']
            if 'pad' in conv_layer['convolution_param']:
                block['pad'] = conv_layer['convolution_param']['pad']
            else:
                block['pad']=0
            last_layer = conv_layer 
            m_conv_layer = lmap[conv_layer['name']] 
            if i+2 < layer_num and layers[i+1]['type'] == 'BatchNorm' and layers[i+2]['type'] == 'Scale':
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                print(i+2,layers[i+2]['name'], layers[i+2]['type'])
                block['batch_normalize'] = '1'
                bn_layer = layers[i+1]
                scale_layer = layers[i+2]
                last_layer = scale_layer
                m_scale_layer = lmap[scale_layer['name']]
                m_bn_layer = lmap[bn_layer['name']]
                wdata += list(m_scale_layer.blobs[1].data)  ## conv_bias <- sc_beta
                wdata += list(m_scale_layer.blobs[0].data)  ## bn_scale  <- sc_alpha
                wdata += (np.array(m_bn_layer.blobs[0].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_mean <- bn_mean/bn_scale
                wdata += (np.array(m_bn_layer.blobs[1].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_var  <- bn_var/bn_scale
                i = i + 2
            else:
                wdata += list(m_conv_layer.blobs[1].data)   ## conv_bias
                block['batch_normalize'] = '0'
            wdata += list(m_conv_layer.blobs[0].data)       ## conv_weights
            
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id[top] = len(blocks)   
                blocks.append(block)          
                i = i + 1
            elif i+1 < layer_num and layers[i+1]['type'] == 'PReLU':
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                prelu_layer = lmap[act_layer['name']]
                wdata += list(prelu_layer.blobs[0].data)
                block['activation'] = 'Prelu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            else:
                block['activation'] = 'linear'
                top = last_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
            i = i + 1
######################################### myself
        elif layer['type'] == 'DepthwiseConvolution':
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            #assert(i+1 < layer_num and layers[i+1]['type'] == 'BatchNorm')
            #assert(i+2 < layer_num and layers[i+2]['type'] == 'Scale')
            conv_layer = layers[i]
            block = OrderedDict()
            block['type'] = 'convolutional'  #'DepthwiseConvolution'
            block['filters'] = conv_layer['convolution_param']['num_output']
            
######################################### myself
            if 'kernel_size' in conv_layer['convolution_param']:
                block['size'] = conv_layer['convolution_param']['kernel_size']
                
            elif 'kernel_h' in conv_layer['convolution_param'] and 'kernel_w' in conv_layer['convolution_param']:
                block['size_h'] = conv_layer['convolution_param']['kernel_h']
                block['size_w'] = conv_layer['convolution_param']['kernel_w']
#########################################
            
#            block['size'] = conv_layer['convolution_param']['kernel_size']
            if 'stride' in conv_layer['convolution_param']:
                block['stride'] = conv_layer['convolution_param']['stride']
            else:
                block['stride'] = '1'  # conv_layer['convolution_param']['stride']
            if 'pad' in conv_layer['convolution_param']:
                block['pad'] = conv_layer['convolution_param']['pad']
            else:
                block['pad']=0
                
            #########  myself
            if 'group' in conv_layer['convolution_param']:
                block['groups'] = conv_layer['convolution_param']['group']
                
            #########
            last_layer = conv_layer 
            m_conv_layer = lmap[conv_layer['name']] 
            if i+2 < layer_num and layers[i+1]['type'] == 'BatchNorm' and layers[i+2]['type'] == 'Scale':
                print (i+1,layers[i+1]['name'], layers[i+1]['type'])
                print (i+2,layers[i+2]['name'], layers[i+2]['type'])
                block['batch_normalize'] = '1'
                bn_layer = layers[i+1]
                scale_layer = layers[i+2]
                last_layer = scale_layer
                m_scale_layer = lmap[scale_layer['name']]
                m_bn_layer = lmap[bn_layer['name']]
                wdata += list(m_scale_layer.blobs[1].data)  ## conv_bias <- sc_beta
                wdata += list(m_scale_layer.blobs[0].data)  ## bn_scale  <- sc_alpha
                wdata += (np.array(m_bn_layer.blobs[0].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_mean <- bn_mean/bn_scale
                wdata += (np.array(m_bn_layer.blobs[1].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_var  <- bn_var/bn_scale
                i = i + 2
            else:
                wdata += list(m_conv_layer.blobs[1].data)   ## conv_bias
                block['batch_normalize'] = '0'
            wdata += list(m_conv_layer.blobs[0].data)       ## conv_weights
            
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            elif i+1 < layer_num and layers[i+1]['type'] == 'PReLU':
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                prelu_layer = lmap[act_layer['name']]
                wdata += list(prelu_layer.blobs[0].data)
                block['activation'] = 'Prelu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            else:
                block['activation'] = 'linear'
                top = last_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
            i = i + 1
###################################################---adding
        elif layer['type'] == 'BatchNorm' and i+1 < layer_num and layers[i+1]['type'] == 'Scale':
                assert(layer_id[layer['bottom']] == len(blocks)-1)
                block = OrderedDict()
                block['type'] = 'batchnorm'
                print (i,layers[i]['name'], layers[i]['type'])
                print (i+1,layers[i+1]['name'], layers[i+1]['type'])
#                block['batch_normalize'] = '1'
                bn_layer = layers[i]
                scale_layer = layers[i+1]
                last_layer = scale_layer
                m_scale_layer = lmap[scale_layer['name']]
                m_bn_layer = lmap[bn_layer['name']]
                wdata += list(m_scale_layer.blobs[1].data)  ## conv_bias <- sc_beta
                wdata += list(m_scale_layer.blobs[0].data)  ## bn_scale  <- sc_alpha
                wdata += (np.array(m_bn_layer.blobs[0].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_mean <- bn_mean/bn_scale
                wdata += (np.array(m_bn_layer.blobs[1].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_var  <- bn_var/bn_scale
                top = layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 2
###################################################---adding
        elif layer['type'] == 'Pooling':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            #---adding lp
#            if layer_id[layer['bottom']] != len(blocks)-1:
#                block = OrderedDict()
#                block['type'] = 'route'
#                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
#                print("route", layer_id[layer['bottom']], len(blocks), block['layers'])
#                blocks.append(block)
            #---adding lp
            block = OrderedDict()
            if layer['pooling_param']['pool'] == 'AVE':
                block['type'] = 'avgpool'
            elif layer['pooling_param']['pool'] == 'MAX':
                block['type'] = 'maxpool'
                block['size'] = layer['pooling_param']['kernel_size']
                block['stride'] = layer['pooling_param']['stride']
#                if layer['pooling_param'].has_key('pad'):
                if 'pad' in layer['pooling_param']:
                    pad = int(layer['pooling_param']['pad'])
                    if pad > 0:
                        block['pad'] = '1'
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1
        elif layer['type'] == 'Eltwise':
            bottoms = layer['bottom']
            bottom1 = layer_id[bottoms[0]] - len(blocks)
            bottom2 = layer_id[bottoms[1]] - len(blocks)
            assert(bottom1 == -1 or bottom2 == -1)
            from_id = bottom2 if bottom1 == -1 else bottom1
            block = OrderedDict()
            block['type'] = 'shortcut'
            block['from'] = str(from_id)
            assert(i+1 < layer_num)
            if layers[i+1]['type'] == 'ReLU':
                block['activation'] = 'relu'
                top = layers[i + 1]['top']
                i = i + 2
            elif layers[i+1]['type'] == 'PReLU':
                block['activation'] = 'Prelu'
                top = layers[i + 1]['top']
                i = i + 2
            else:
                top = layers[i]['top']
                i = i + 1
            layer_id[top] = len(blocks)
            blocks.append(block)

        elif layer['type'] == 'InnerProduct':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'connected'
            block['output'] = layer['inner_product_param']['num_output']
            m_fc_layer = lmap[layer['name']]
            
################################ myself before 12.06.2018
#            if len(m_fc_layer.blobs)>1:
#                wdata += list(m_fc_layer.blobs[0].data)       ## fc_weights
#                wdata += list(m_fc_layer.blobs[1].data)       ## fc_bias
#            elif len(m_fc_layer.blobs) ==1 :
#                wdata += list(m_fc_layer.blobs[0].data)
################################# myself
################---adding  by ylsun on 12.06.2018
            if i+2 < layer_num and layers[i+1]['type'] == 'BatchNorm' and layers[i+2]['type'] == 'Scale':
                print (i+1,layers[i+1]['name'], layers[i+1]['type'])
                print (i+2,layers[i+2]['name'], layers[i+2]['type'])
                block['batch_normalize'] = '1'
                bn_layer = layers[i+1]
                scale_layer = layers[i+2]
                last_layer = scale_layer
                m_scale_layer = lmap[scale_layer['name']]
                m_bn_layer = lmap[bn_layer['name']]
                wdata += list(m_scale_layer.blobs[1].data)     ## conv_bias <- sc_beta  ????????TODO
#                wdata += list(m_fc_layer.blobs[0].data)       ## fc_weights
                wdata += list(m_scale_layer.blobs[0].data)     ## bn_scale  <- sc_alpha
                wdata += (np.array(m_bn_layer.blobs[0].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_mean <- bn_mean/bn_scale
                wdata += (np.array(m_bn_layer.blobs[1].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_var  <- bn_var/bn_scale
                i = i + 2
            else:
                wdata += list(m_fc_layer.blobs[1].data)   ## fc_bias
                block['batch_normalize'] = '0'
            wdata += list(m_fc_layer.blobs[0].data)       ## fc_weights
################---adding
       
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
#                i = i + 2
                i = i + 1
            if i+1 < layer_num and layers[i+1]['type'] == 'PReLU':
                act_layer = layers[i+1]
                block['activation'] = 'Prelu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
#                i = i + 2
                i = i + 1
            else:
                block['activation'] = 'linear'
                top = layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
                
        elif layer['type'] == 'Softmax':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'softmax'
            block['groups'] = 1
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

        elif layer['type'] == 'Concat':
            block = OrderedDict()
            block['type'] = 'route'
            route_layers=""
            print(layer['bottom'])
            for bottom_num in layer['bottom']:
                layer_id[bottom_num] == len(blocks)-1
                route_layers += str(layer_id[bottom_num] - len(blocks))+","
            route_layers = route_layers[0:-1]
            block['layers'] = route_layers
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1
            
        else:
            print('unknown type %s' % layer['type'])
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = layer['type']
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            block = OrderedDict()
            block['type'] = layer['type']
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)

            i = i + 1

    print('done')
    return blocks, np.array(wdata)

def prototxt2cfg(protofile):
    net_info = parse_prototxt(protofile)
    props = net_info['props']

    blocks = []
    block = OrderedDict()
    block['type'] = 'net' 
#    if props.has_key('input_shape'): 
    if 'input_shape' in props:
        block['batch'] = props['input_shape']['dim'][0]
        block['channels'] = props['input_shape']['dim'][1]
        block['height'] = props['input_shape']['dim'][2]
        block['width'] = props['input_shape']['dim'][3]
    else:
        block['batch'] = props['input_dim'][0]
        block['channels'] = props['input_dim'][1]
        block['height'] = props['input_dim'][2]
        block['width'] = props['input_dim'][3]
#    if props.has_key('mean_file'):
    if 'mean_file' in props:
        block['mean_file'] = props['mean_file']
    blocks.append(block)

    layers = net_info['layers']
    layer_num = len(layers)
    i = 0 # layer id
    layer_id = dict()
    layer_id[props['input']] = 0
    while i < layer_num:
        layer = layers[i]
        print(i,layer['name'], layer['type'])
        if layer['type'] == 'Convolution':
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            conv_layer = layers[i]
            block = OrderedDict()
            block['type'] = 'convolutional'
            block['filters'] = conv_layer['convolution_param']['num_output']
            block['size'] = conv_layer['convolution_param']['kernel_size']
            block['stride'] = '1'
#            if conv_layer['convolution_param'].has_key('stride'):
            if 'stride' in  conv_layer['convolution_param']:
                block['stride'] = conv_layer['convolution_param']['stride']
            block['pad'] = '1'
            last_layer = conv_layer 
            if i+2 < layer_num and layers[i+1]['type'] == 'BatchNorm' and layers[i+2]['type'] == 'Scale':
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                print( i+2,layers[i+2]['name'], layers[i+2]['type'])
                block['batch_normalize'] = '1'
                bn_layer = layers[i+1]
                scale_layer = layers[i+2]
                last_layer = scale_layer
                i = i + 2
            
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                print (i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            elif i+1 < layer_num and layers[i+1]['type'] == 'PReLU':
                print (i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                block['activation'] = 'Prelu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            else:
                block['activation'] = 'linear'
                top = last_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
            i = i + 1
######################################## myself
        elif layer['type'] == 'DepthwiseConvolution':
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            conv_layer = layers[i]
            block = OrderedDict()
            block['type'] = 'convolutional'
            block['filters'] = conv_layer['convolution_param']['num_output']
            ################myself
            if 'kernel_size' in conv_layer['convolution_param']:
                block['size'] = conv_layer['convolution_param']['kernel_size']
                
            elif 'kernel_h' in conv_layer['convolution_param'] and 'kernel_w' in conv_layer['convolution_param']:
                block['size_h'] = conv_layer['convolution_param']['kernel_h']
                block['size_w'] = conv_layer['convolution_param']['kernel_w']
#            block['size'] = conv_layer['convolution_param']['kernel_size']
            ################# 
            block['stride'] = '1'
#            if conv_layer['convolution_param'].has_key('stride'):
            if 'stride' in  conv_layer['convolution_param']:
                block['stride'] = conv_layer['convolution_param']['stride']
            block['pad'] = '1'
            last_layer = conv_layer 
            if i+2 < layer_num and layers[i+1]['type'] == 'BatchNorm' and layers[i+2]['type'] == 'Scale':
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                print( i+2,layers[i+2]['name'], layers[i+2]['type'])
                block['batch_normalize'] = '1'
                bn_layer = layers[i+1]
                scale_layer = layers[i+2]
                last_layer = scale_layer
                i = i + 2
            
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                print (i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            elif i+1 < layer_num and layers[i+1]['type'] == 'PReLU':
                print (i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                block['activation'] = 'Prelu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            else:
                block['activation'] = 'linear'
                top = last_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
            i = i + 1
########################################
        elif layer['type'] == 'Pooling':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            if layer['pooling_param']['pool'] == 'AVE':
                block['type'] = 'avgpool'
            elif layer['pooling_param']['pool'] == 'MAX':
                block['type'] = 'maxpool'
                block['size'] = layer['pooling_param']['kernel_size']
                block['stride'] = layer['pooling_param']['stride']
#                if layer['pooling_param'].has_key('pad'):
                if 'pad' in layer['pooling_param']:
                    pad = int(layer['pooling_param']['pad'])
                    if pad > 0:
                        block['pad'] = '1'
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1
        elif layer['type'] == 'Eltwise':
            bottoms = layer['bottom']
            bottom1 = layer_id[bottoms[0]] - len(blocks)
            bottom2 = layer_id[bottoms[1]] - len(blocks)
            assert(bottom1 == -1 or bottom2 == -1)
            from_id = bottom2 if bottom1 == -1 else bottom1
            block = OrderedDict()
            block['type'] = 'shortcut'
            block['from'] = str(from_id)
            assert(i+1 < layer_num)
            if layers[i+1]['type'] == 'ReLU':
                block['activation'] = 'relu'
            elif layers[i+1]['type'] == 'PReLU':
                block['activation'] = 'Prelu'
            top = layers[i+1]['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 2
        elif layer['type'] == 'InnerProduct':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'connected'
            block['output'] = layer['inner_product_param']['num_output']
            
################### ---adding by 12.06.2018
            if i+2 < layer_num and layer[i+1]['type'] == 'BatchNorm' and layers[i+2]['type'] == 'Scale':
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                print( i+2,layers[i+2]['name'], layers[i+2]['type'])
                block['batch_normalize'] = '1'
                bn_layer = layers[i+1]
                scale_layer = layers[i+2]
                last_layer = scale_layer
                i = i + 2 
################### ---adding
            
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                print (i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 2
            elif i+1 < layer_num and layers[i+1]['type'] == 'PReLU':
                print (i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                block['activation'] = 'Prelu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 2
            else:
                block['activation'] = 'linear'
                top = layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
        elif layer['type'] == 'Softmax':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'softmax'
            block['groups'] = 1
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1
        else:
            print('unknown type %s' % layer['type'])
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            block = OrderedDict()
            block['type'] = layer['type']
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

    print('done') 
    return blocks


def save_weights(data, weightfile):
    print('Save to ', weightfile)
    wsize = data.size
    weights = np.zeros((wsize+4,), dtype=np.int32)
    ## write info 
    weights[0] = 0
    weights[1] = 1
    weights[2] = 0      ## revision
    weights[3] = 0      ## net.seen
    weights.tofile(weightfile)
    weights = np.fromfile(weightfile, dtype=np.float32)
    weights[4:] = data
    weights.tofile(weightfile)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 5:
        print('try:')
        print('  python caffe2darknet.py ResNet-50-deploy.prototxt ResNet-50-model.caffemodel ResNet-50-model.cfg ResNet-50-model.weights')
        exit()
        
#    sys.argv[0] = 'caffe2darknet.py'
#    sys.argv[1] = '/home/zongqing/Desktop/caffe_darknet/scripts/model_convert/MobileFaceNet_deploy.prototxt'
#    sys.argv[2] = '/home/zongqing/Desktop/caffe_darknet/scripts/model_convert/MobileFaceNet.caffemodel'
#    sys.argv[3] = 'mobileFacenet_test.cfg'
#    sys.argv[4] = 'mobileFacenet_test.weights'
#    
#    protofile = sys.argv[1]
#    caffemodel = sys.argv[2]
#    cfgfile = sys.argv[3]
#    weightfile = sys.argv[4]
 
    protofile = '/home/hup/project/ReID/reidS/pytorch2caffe_MGN/MGN.prototxt'
    caffemodel = '/home/hup/project/ReID/reidS/pytorch2caffe_MGN/MGN.caffemodel'
    cfgfile = '/home/hup/project/ReID/reidS/pytorch2caffe_MGN/MGN.cfg'
    weightfile = '/home/hup/project/ReID/reidS/pytorch2caffe_MGN/MGN.weights'
    #protofile = '../models/pose/body_25/pose_deploy.prototxt'
    #caffemodel = '../models/pose/body_25/pose_iter_584000.caffemodel'
    #cfgfile = '/home/hup/project/pytestMnist/body_25/pose_3.cfg'
    #weightfile = '/home/hup/project/pytestMnist/body_25/pose3_58400.weights'
    
    blocks, data = caffe2darknet(protofile, caffemodel)
    
    save_weights(data, weightfile)
    save_cfg(blocks, cfgfile)
    
    print_cfg(blocks)
    print_cfg_nicely(blocks)
    
    
    
