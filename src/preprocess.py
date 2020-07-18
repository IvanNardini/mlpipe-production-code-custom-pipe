



# Training function



#Scoring function
def model_scorer(X, model, index):
    '''
    Score new data with onnx
    :params: X, model, index
    :return: list
    '''
    sess = rt.InferenceSession(model)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    row_to_score = pd.DataFrame([X.iloc[index]])

    score = np.array(row_to_score, dtype=np.float32)
    predictions_onnx = sess.run([label_name], {input_name: score})
    return predictions_onnx[0]




    
