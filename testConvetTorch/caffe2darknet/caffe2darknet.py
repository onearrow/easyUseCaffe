#!/home/ubuntu/anaconda2/bin/python -f 

# Python 2/3 compatibility
from __future__ import print_function

from collections import OrderedDict
from cfg import *
from prototxt import *
import numpy as np

def caffe2darknet(protofile, caffemodel):
    import topological_sort
    # build DAG and sort
    net_info = parse_prototxt(protofile)
    props = net_info['props']
    org_layers = net_info['layers']

    layer_activation_by_name = {}
    for layer in org_layers:
        layer_name = layer['name']
        layer_type = layer['type']
        bottom_name = layer['bottom']
        if layer_type in ('ReLU', 'PReLU'):
            layer_activation_by_name[bottom_name] = layer_name

    org_layer_id_by_name = OrderedDict()
    graph = {}
    for i in range(len(org_layers)):
        layer = org_layers[i]
        layer_name = layer['name']
        layer_type = layer['type']
        bottom_name = layer['bottom']
        org_layer_id_by_name[layer_name] = i
        if not isinstance(bottom_name, list):
            bottom_name = [bottom_name]
        for b_name in bottom_name:
            if layer_type in ('Pooling', 'Convolution', 'Concat'):
                # change b_name
                if b_name in layer_activation_by_name:
                    b_name = layer_activation_by_name[b_name]
            top_layers = graph.get(b_name, [])
            top_layers.append(layer_name)
            graph[b_name] = top_layers
    if 'image' in graph:
        del graph['image']
    sorted_layer_names = topological_sort.topological_sort(graph)
    for s in sorted_layer_names:
        print(s)

    # load weights and build name -> weights map
    layer_weights_by_name = {}

    if True:
        model = parse_caffemodel(caffemodel)
        model_layers = model.layer
        if len(model_layers) == 0:
            print('Using V1LayerParameter')
            model_layers = model.layers
        
        for layer in model_layers:
            layer_weights_by_name[layer.name] = layer

    # 
    weights_data = [] # *.weights
    cfg_blocks = []  # *.cfg
    
    block = OrderedDict()
    block['type'] = 'net'
    if 'input_shape' in props:
        block['width'] = props['input_shape']['dim'][3]
        block['height'] = props['input_shape']['dim'][2]
        block['batch'] = props['input_shape']['dim'][0]
        block['channels'] = props['input_shape']['dim'][1]
    else:
        block['width'] = props['input_dim'][3]
        block['height'] = props['input_dim'][2]
        block['batch'] = props['input_dim'][0]
        block['channels'] = props['input_dim'][1]
    if int(block['width']) < 10:	#100 modify to 10
        block['width'] = 200
    if int(block['height']) < 10:	#100 modify to 10
        block['height'] = 200
    if 'mean_file' in props:
        block['mean_file'] = props['mean_file']
    cfg_blocks.append(block)

    layer_num = len(sorted_layer_names)
    layers = []
    for layer_name in sorted_layer_names:
        org_id = org_layer_id_by_name[layer_name]
        layers.append(org_layers[org_id])
    del org_layer_id_by_name
    i = 0 # layer id
    layer_id_by_name = dict()
    layer_id_by_name[props['input']] = 0
    while i < layer_num:
        layer = layers[i]
        layer_name = layer['name']
        layer_type = layer['type']
        print(i, layer_name, layer_type)
        block = OrderedDict()
        block['name'] =  layer_name

        if layer_type == 'Convolution':
            if layer_id_by_name[layer['bottom']] != len(cfg_blocks)-1:
                a_block = OrderedDict()
                a_block['name'] =  layer_name
                # a_block['bottom'] = layer['bottom']
                # a_block['bottom layer_id'] = layer_id_by_name[layer['bottom']]
                # a_block['gold_bottom'] = len(cfg_a_blocks) - 1
                a_block['type'] = 'route'
                a_block['layers'] = str(layer_id_by_name[layer['bottom']] - len(cfg_blocks))
                cfg_blocks.append(a_block)
            #assert(i+1 < layer_num and layers[i+1]['type'] == 'BatchNorm')
            #assert(i+2 < layer_num and layers[i+2]['type'] == 'Scale')
            
            block['type'] = 'convolutional'
            block['batch_normalize'] = '0'
            block['filters'] = layer['convolution_param']['num_output']
            block['size'] = layer['convolution_param']['kernel_size']
            block['stride'] = layer['convolution_param'].get('stride', 1)
            block['pad'] = layer['convolution_param'].get('pad', 0)
            last_layer = layer 
            weights_conv_layer = layer_weights_by_name[layer_name] 
            if i+2 < layer_num and layers[i+1]['type'] == 'BatchNorm' and layers[i+2]['type'] == 'Scale':
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                print(i+2,layers[i+2]['name'], layers[i+2]['type'])
                block['batch_normalize'] = '1'
                bn_layer = layers[i+1]
                scale_layer = layers[i+2]
                last_layer = scale_layer
                weights_scale_layer = layer_weights_by_name[scale_layer['name']]
                weights_bn_layer = layer_weights_by_name[bn_layer['name']]
                weights_data += list(weights_scale_layer.blobs[1].data)  ## conv_bias <- sc_beta
                weights_data += list(weights_scale_layer.blobs[0].data)  ## bn_scale  <- sc_alpha
                weights_data += (np.array(weights_bn_layer.blobs[0].data) / weights_bn_layer.blobs[2].data[0]).tolist()  ## bn_mean <- bn_mean/bn_scale
                weights_data += (np.array(weights_bn_layer.blobs[1].data) / weights_bn_layer.blobs[2].data[0]).tolist()  ## bn_var  <- bn_var/bn_scale
                i = i + 2
            else:
                weights_data += list(weights_conv_layer.blobs[1].data)   ## conv_bias
            weights_data += list(weights_conv_layer.blobs[0].data)       ## conv_weights
            
            if i + 1 < layer_num and layers[i + 1]['type'] in ('ReLU', 'PReLU'):
                print(i+1,layers[i+1]['name'], layers[i+1]['type'])
                act_layer = layers[i+1]
                if layers[i + 1]['type'] == 'ReLU':
                    block['activation'] = 'relu'
                else:
                    block['activation'] = 'Prelu'	#relie modify to Prelu
                top = act_layer['top']
                layer_id_by_name[top] = len(cfg_blocks)
                cfg_blocks.append(block)
                i = i + 1
            else:
                block['activation'] = 'linear'
                top = last_layer['top']
                layer_id_by_name[top] = len(cfg_blocks)
                cfg_blocks.append(block)
            i = i + 1
        elif layer_type == 'Pooling':
            assert(layer_id_by_name[layer['bottom']] == len(cfg_blocks)-1)
            
            if layer['pooling_param']['pool'] == 'AVE':
                block['type'] = 'avgpool'
            elif layer['pooling_param']['pool'] == 'MAX':
                block['type'] = 'maxpool'
                block['size'] = layer['pooling_param']['kernel_size']
                block['stride'] = layer['pooling_param'].get('stride', 1)
                if 'pad' in layer['pooling_param']:
                    pad = int(layer['pooling_param']['pad'])
                    if pad > 0:
                        block['pad'] = '1'
            top = layer['top']
            layer_id_by_name[top] = len(cfg_blocks)
            cfg_blocks.append(block)
            i = i + 1
        elif layer_type == 'Eltwise':
            bottoms = layer['bottom']
            bottom1 = layer_id_by_name[bottoms[0]] - len(cfg_blocks)
            bottom2 = layer_id_by_name[bottoms[1]] - len(cfg_blocks)
            assert(bottom1 == -1 or bottom2 == -1)
            from_id = bottom2 if bottom1 == -1 else bottom1
            
            block['type'] = 'shortcut'
            block['from'] = str(from_id)
            assert(i+1 < layer_num and layers[i+1]['type'] == 'ReLU')
            block['activation'] = 'relu'
            top = layers[i+1]['top']
            layer_id_by_name[top] = len(cfg_blocks)
            cfg_blocks.append(block)
            i = i + 2
        elif layer_type == 'InnerProduct':
            assert(layer_id_by_name[layer['bottom']] == len(cfg_blocks)-1)
            
            block['type'] = 'connected'
            block['output'] = layer['inner_product_param']['num_output']
            weights_fc_layer = layer_weights_by_name[layer_name]
            weights_data += list(weights_fc_layer.blobs[1].data)       ## fc_bias
            weights_data += list(weights_fc_layer.blobs[0].data)       ## fc_weights
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id_by_name[top] = len(cfg_blocks)
                cfg_blocks.append(block)
                i = i + 2
            else:
                block['activation'] = 'linear'
                top = layer['top']
                layer_id_by_name[top] = len(cfg_blocks)
                cfg_blocks.append(block)
                i = i + 1
        elif layer_type == 'Softmax':
            assert(layer_id_by_name[layer['bottom']] == len(cfg_blocks)-1)
            
            block['type'] = 'softmax'
            block['groups'] = 1
            top = layer['top']
            layer_id_by_name[top] = len(cfg_blocks)
            cfg_blocks.append(block)
            i = i + 1
        elif layer_type == 'Concat':
            block['type'] = 'route'
            top = layer['top']
            concat_layers = []
            for b_name in layer['bottom']:
                id_str = str(layer_id_by_name[b_name] - len(cfg_blocks))
                concat_layers.append(id_str)
            block['layers'] = ','.join(concat_layers)
            # print(block)
            layer_id_by_name[top] = len(cfg_blocks)
            cfg_blocks.append(block)
            i = i + 1
        else:
            print('unknown type %s' % layer_type)
            if layer_id_by_name[layer['bottom']] != len(cfg_blocks)-1:
                block['type'] = 'WTF'
                block['layers'] = str(layer_id_by_name[layer['bottom']] - len(cfg_blocks))
                cfg_blocks.append(block)
            
            block['type'] = layer_type
            top = layer['top']
            layer_id_by_name[top] = len(cfg_blocks)
            cfg_blocks.append(block)

            i = i + 1

    print('done' )
    return cfg_blocks, np.array(weights_data)

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
    
    if len(sys.argv) != 1:
        sys.argv = [
            sys.argv[0],
            'd:/__svn_pool/openpose/models/pose/body_25/pose_deploy.prototxt',
            'd:/__svn_pool/openpose/models/pose/body_25/pose_iter_584000.caffemodel',
            '../../jing-pose/pose_deploy.cfg',
            '../../jing-pose/pose_deploy.weights',
        ]
    #protofile = sys.argv[1]
    #caffemodel = sys.argv[2]
    #cfgfile = sys.argv[3]
    #weightfile = sys.argv[4]
    protofile = '/home/hup/project/ReID/reidS/pytorch2caffe_MGN/MGN.prototxt'
    caffemodel = '/home/hup/project/ReID/reidS/pytorch2caffe_MGN/MGN.caffemodel'
    cfgfile = '/home/hup/project/ReID/reidS/pytorch2caffe_MGN/MGN_dark.cfg'
    weightfile = '/home/hup/project/ReID/reidS/pytorch2caffe_MGN/MGN_dark.weights'
    cfg_blocks, data = caffe2darknet(protofile, caffemodel)
    
    save_cfg(cfg_blocks, cfgfile)
    
    save_weights(data, weightfile)
    
    # print_cfg(cfg_blocks)
    # print_cfg_nicely(cfg_blocks)
