from sklearn.model_selection import KFold
import numpy as np
import sympy as sp
import ast

##########  This algorithm is only a case study of a three-source fusion task  #############

# Generate Sugeno-lambda FM
def get_Sugeno(g1,g2,g3):
    lambda_ori = sp.Symbol('x')
    f_original = (1 + lambda_ori * g1) * (1 + lambda_ori * g2) * (1 + lambda_ori * g3) - (lambda_ori + 1)
    lambda_ori = sp.Symbol('x')
    f = sp.expand(f_original)
    lambda_ori = sp.solve(f)

    g_lambda = [x for x in lambda_ori if x > -1 and x != float(0)]

    g12 = g1 + g2 + g_lambda[0] * g1 * g2
    g13 = g1 + g3 + g_lambda[0] * g1 * g3
    g23 = g2 + g3 + g_lambda[0] * g2 * g3

    g_group = np.array([[g1,g2,g3,g12,g13,g23,1]]).flatten()    

    return g_group

# Generate Decomposable FM
def get_decompose_fuzzy_measure(g1,g2,g3) :

    g12 = min(g1+g2,1)
    g13 = min(g1+g3,1)
    g23 = min(g2+g3,1)
    g123 = min(g1+g2+g3,1)

    g_group = np.array([[g1,g2,g3,g12,g13,g23,g123]]).flatten()

    return g_group

# Choquet FI
def get_ChI(output_shape, output_texture, output_color, g_measure) :
    input_ChI = np.zeros((3, len(output_shape)))
    input_ChI[0,:] = output_shape
    input_ChI[1,:] = output_texture
    input_ChI[2,:] = output_color

    ChI_output = np.zeros((len(output_shape)))
   
    for i in range(len(output_shape)) :

        h = np.array([input_ChI[0,i], input_ChI[1,i], input_ChI[2,i]])
        h_dec = np.argsort(h)[::-1]

        if np.all(h_dec == [0,1,2]) :
            ChI_output[i] = g_measure[0]*h[h_dec[0]] + (g_measure[3] - g_measure[0])*h[h_dec[1]] + (g_measure[6] - g_measure[3])*h[h_dec[2]]
        elif np.all(h_dec == [0,2,1]) :
            ChI_output[i] = g_measure[0]*h[h_dec[0]] + (g_measure[4] - g_measure[0])*h[h_dec[1]] + (g_measure[6] - g_measure[4])*h[h_dec[2]]
        elif np.all(h_dec == [1,0,2]) :
            ChI_output[i] = g_measure[1]*h[h_dec[0]] + (g_measure[3] - g_measure[1])*h[h_dec[1]] + (g_measure[6] - g_measure[3])*h[h_dec[2]]
        elif np.all(h_dec == [1,2,0]) :
            ChI_output[i] = g_measure[1]*h[h_dec[0]] + (g_measure[5] - g_measure[1])*h[h_dec[1]] + (g_measure[6] - g_measure[5])*h[h_dec[2]]
        elif np.all(h_dec == [2,0,1]) :
            ChI_output[i] = g_measure[2]*h[h_dec[0]] + (g_measure[4] - g_measure[2])*h[h_dec[1]] + (g_measure[6] - g_measure[4])*h[h_dec[2]]
        elif np.all(h_dec == [2,1,0]) :
            ChI_output[i] = g_measure[2]*h[h_dec[0]] + (g_measure[5] - g_measure[2])*h[h_dec[1]] + (g_measure[6] - g_measure[5])*h[h_dec[2]]  
        
        # g_measure[0] = g1, g_measure[1] = g2, g_measure[2] = g3
        # g_measure[3] = g12, g_measure[4] = g13, g_measure[5] = g23, g_measure[6] = g123

    return ChI_output

# Sugeno FI
def get_SI(output_shape, output_texture, output_color, g_measure) :
    input_SI = np.zeros((3, len(output_shape)))
    input_SI[0,:] = output_shape
    input_SI[1,:] = output_texture
    input_SI[2,:] = output_color

    SI_output = np.zeros((len(output_shape)))
   
    for i in range(len(output_shape)) :

        h = np.array([input_SI[0,i], input_SI[1,i], input_SI[2,i]])
        h_dec = np.argsort(h)[::-1]

        if np.all(h_dec == [0,1,2]) :
            SI_output[i] = max(min(g_measure[0],h[h_dec[0]]),min(g_measure[3],h[h_dec[1]]),min(g_measure[6],h[h_dec[2]]))
        elif np.all(h_dec == [0,2,1]) :
            SI_output[i] = max(min(g_measure[0],h[h_dec[0]]),min(g_measure[4],h[h_dec[1]]),min(g_measure[6],h[h_dec[2]]))
        elif np.all(h_dec == [1,0,2]) :
            SI_output[i] = max(min(g_measure[1],h[h_dec[0]]),min(g_measure[3],h[h_dec[1]]),min(g_measure[6],h[h_dec[2]]))
        elif np.all(h_dec == [1,2,0]) :
            SI_output[i] = max(min(g_measure[1],h[h_dec[0]]),min(g_measure[5],h[h_dec[1]]),min(g_measure[6],h[h_dec[2]]))
        elif np.all(h_dec == [2,0,1]) :
            SI_output[i] = max(min(g_measure[2],h[h_dec[0]]),min(g_measure[4],h[h_dec[1]]),min(g_measure[6],h[h_dec[2]]))
        elif np.all(h_dec == [2,1,0]) :
            SI_output[i] = max(min(g_measure[2],h[h_dec[0]]),min(g_measure[5],h[h_dec[1]]),min(g_measure[6],h[h_dec[2]])) 

        # g_measure[0] = g1, g_measure[1] = g2, g_measure[2] = g3
        # g_measure[3] = g12, g_measure[4] = g13, g_measure[5] = g23, g_measure[6] = g123

        # SI_output[i] = SI_output[i] * max(input_SI[0,i], input_SI[1,i], input_SI[2,i])

    return SI_output

# Weighted Average
def get_WA(output_shape,output_texture,output_color,WA_weight):
    input = np.zeros((3, len(output_shape)))
    input[0,:] = output_shape
    input[1,:] = output_texture
    input[2,:] = output_color

    WA_weight = WA_weight / np.sum(WA_weight)

    WA_output = np.zeros((len(output_shape)))

    for i in range(len(output_shape)) :

        WA_output[i] = input[0,i] * WA_weight[0] + input[1,i] * WA_weight[1] + input[2,i] * WA_weight[2]


    return WA_output

# Load Data
def data_loader(path) :
    data_list = []

    with open(path, 'r') as f:
        content = f.read()

    data_sections = content.strip().split('\n\n\n')

    for section in data_sections:
        lines = section.splitlines()
        data_dict = {}
        
        for i in range(len(lines)):
            if lines[i].startswith("label:"):
                label_data = ast.literal_eval(lines[i].split(":", 1)[1].strip())
                data_dict["label"] = label_data 
            elif lines[i].startswith("output_shape:"):
                shape_data = ast.literal_eval(lines[i + 1].strip())
                data_dict["output_shape"] = shape_data
            elif lines[i].startswith("output_texture:"):
                texture_data = ast.literal_eval(lines[i + 1].strip())
                data_dict["output_texture"] = texture_data
            elif lines[i].startswith("output_color:"):
                color_data = ast.literal_eval(lines[i + 1].strip())
                data_dict["output_color"] = color_data
        
        data_list.append(data_dict)
    
    return data_list

# Apply K_Fold
def K_Fold(data_list, num_fold=10) :

    # Load data
    output_shapes = np.array(data_list[0]['output_shape'])
    output_textures = np.array(data_list[0]['output_texture'])
    output_colors = np.array(data_list[0]['output_color'])
    labels = np.array(data_list[0]['label'])

    # Create K-fold instance
    kf = KFold(n_splits=num_fold , shuffle=True, random_state=42)

    sum_acc_WA = 0
    sum_acc_Decompose = 0
    sum_acc_Sugeno = 0
    sum_acc_Average = 0
    sum_acc_SI_Decompose = 0
    sum_acc_SI_Sugeno = 0

    list_acc_shape = []
    list_acc_texture = []
    list_acc_color = []

    # Apply 10-fold 
    for fold_idx, (train_index, test_index) in enumerate(kf.split(output_shapes)):
        print(f"Processing Fold {fold_idx + 1}...")

        # Obtain training and test set for each fold
        train_shapes, test_shapes = output_shapes[train_index], output_shapes[test_index]
        train_textures, test_textures = output_textures[train_index], output_textures[test_index]
        train_colors, test_colors = output_colors[train_index], output_colors[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        acc_WA = 0
        acc_Decompose = 0
        acc_Sugeno = 0
        acc_Average = 0
        acc_SI_Decompose = 0
        acc_SI_Sugeno = 0

        acc_shape = 0
        acc_texture = 0
        acc_color = 0

        # Calculate densiteis (weights for weighted average) for each fold on training set
        for i in range(len(train_labels)) :
            output_label = train_labels[i]
            output_shape = train_shapes[i]
            output_texture = train_textures[i]
            output_color = train_colors[i]

            if np.argsort(output_shape)[::-1][0] == output_label :
                acc_shape = acc_shape + 1
            if np.argsort(output_texture)[::-1][0] == output_label :
                acc_texture = acc_texture + 1
            if np.argsort(output_color)[::-1][0] == output_label :
                acc_color = acc_color + 1

        acc_shape = acc_shape / len(train_labels)
        acc_texture = acc_texture / len(train_labels)
        acc_color = acc_color / len(train_labels)

        g1 = acc_shape
        g2 = acc_texture
        g3 = acc_color

        list_acc_shape.append(g1)
        list_acc_texture.append(g2)
        list_acc_color.append(g3)

        # print('shape:',g1,'texture',g2,'color',g3)
        print('Weights:')
        sum_g = np.sum(g1+g2+g3)
        print('shape:',g1/sum_g,'texture',g2/sum_g,'color',g3/sum_g)

        # Test set
        for i in range(len(test_labels)):
            output_label = test_labels[i]
            output_shape = test_shapes[i]
            output_texture = test_textures[i]
            output_color = test_colors[i]

            WA_weight = [g1,g2,g3]
            Average_weight = [1/3,1/3,1/3]

            # Calculate WA and Avg output
            WA_output = get_WA(output_shape,output_texture,output_color,WA_weight)
            Average_output = get_WA(output_shape,output_texture,output_color,Average_weight)

            g_Sugeno = get_Sugeno(g1,g2,g3)
            g_decompose = get_decompose_fuzzy_measure(g1,g2,g3)

            # Calculate FI output
            decompose_output = get_ChI(output_shape, output_texture, output_color, g_decompose)
            Sugeno_output = get_ChI(output_shape, output_texture, output_color, g_Sugeno)
            SI_decompose_output = get_SI(output_shape, output_texture, output_color, g_decompose)
            SI_Sugeno_output = get_SI(output_shape, output_texture, output_color, g_Sugeno)
        
            # Comapre with label
            if np.argsort(WA_output)[::-1][0] == output_label:
                acc_WA += 1
            if np.argsort(Average_output)[::-1][0] == output_label:
                acc_Average += 1
            if np.argsort(decompose_output)[::-1][0] == output_label:
                acc_Decompose += 1
            if np.argsort(Sugeno_output)[::-1][0] == output_label:
                acc_Sugeno += 1       
            if np.argsort(SI_decompose_output)[::-1][0] == output_label:
                acc_SI_Decompose += 1   
            if np.argsort(SI_Sugeno_output)[::-1][0] == output_label:
                acc_SI_Sugeno += 1    

        # Calculate average per fold
        sum_acc_WA = sum_acc_WA + acc_WA
        sum_acc_Average = sum_acc_Average + acc_Average
        sum_acc_Decompose = sum_acc_Decompose + acc_Decompose
        sum_acc_Sugeno = sum_acc_Sugeno + acc_Sugeno
        sum_acc_SI_Decompose = sum_acc_SI_Decompose + acc_SI_Decompose
        sum_acc_SI_Sugeno = sum_acc_SI_Sugeno + acc_SI_Sugeno

        acc_WA = acc_WA / len(test_labels)
        acc_Average = acc_Average / len(test_labels)
        acc_Decompose = acc_Decompose / len(test_labels)
        acc_Sugeno = acc_Sugeno / len(test_labels)
        acc_SI_Decompose = acc_SI_Decompose / len(test_labels)
        acc_SI_Sugeno = acc_SI_Sugeno / len(test_labels)

        # Plot average per fold
        print('Fold',fold_idx+1,'SI_Sugeno_ACC:',acc_SI_Sugeno)
        print('Fold',fold_idx+1,'SI_Decompose_ACC:',acc_SI_Decompose)  
        print('Fold',fold_idx+1,'Sugeno_ACC:',acc_Sugeno)
        print('Fold',fold_idx+1,'Decompose_ACC:',acc_Decompose)   
        print('Fold',fold_idx+1,'Average_ACC:',acc_Average) 
        print('Fold',fold_idx+1,'WA_ACC:',acc_WA) 

        # print(SI_decompose_output)
        # print(SI_Sugeno_output)

    # Calculate overall average
    sum_acc_WA = sum_acc_WA / len(data_list[0]['label'])
    sum_acc_Average = sum_acc_Average / len(data_list[0]['label'])
    sum_acc_Decompose = sum_acc_Decompose / len(data_list[0]['label'])
    sum_acc_Sugeno = sum_acc_Sugeno / len(data_list[0]['label'])
    sum_acc_SI_Decompose = sum_acc_SI_Decompose / len(data_list[0]['label'])
    sum_acc_SI_Sugeno = sum_acc_SI_Sugeno / len(data_list[0]['label'])

    # Plot overall accuracies
    print('Average across all folds')
    print('WA:',sum_acc_WA)
    print('Average:',sum_acc_Average)
    print('Decompose_ACC:',sum_acc_Decompose)
    print('Sugeno_ACC:',sum_acc_Sugeno)
    print('SI_Decompose_ACC:',sum_acc_SI_Decompose)
    print('SI_Sugeno_ACC:',sum_acc_SI_Sugeno)
    print('shape:',np.mean(list_acc_shape))
    print('texture:',np.mean(list_acc_texture))
    print('color:',np.mean(list_acc_color))
 
    # print('Sugeno:',g_Sugeno)
    # print('Decompose:',g_decompose)


if __name__ == "__main__":

    # Load data
    # data_list = data_loader('Combined_dataset.txt') 
    data_list = data_loader('iLab_dataset.txt')

    # Apply K-fold
    K_Fold(data_list,10)
