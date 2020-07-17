

def dumminizer(data, columns_to_dummies, dummies_meta):
    '''
    Generate dummies for nominal variables
    :params: data, columns_to_dummies, dummies_meta
    :return: DataFrame
    '''
    for var in columns_to_dummies:
        cat_names = sorted(dummies_meta[var])
        obs_cat_names = sorted(list(set(data[var].unique())))
        dummies = pd.get_dummies(data[var], prefix=var)
        data = pd.concat([data, dummies], axis=1)
        if obs_cat_names != cat_names: #exception: when label misses 
            cat_miss_labels = ["_".join([var, cat]) for cat in cat_names if cat not in obs_cat_names] #syntetic dummy
            for cat in cat_miss_labels:
                data[cat] = 0 
        data = data.drop(var, 1)
    return data
    
def scaler_trainer(data, output_path):
    '''
    Fit the scaler on predictors
    :params: data, output_path
    :return: scaler
    '''
    
    scaler = MinMaxScaler()
    scaler.fit(data)
    joblib.dump(scaler, output_path)
    return scaler
  
def scaler_trasformer(data, scaler):
    '''
    Trasform the data 
    :params: data, scaler
    :return: DataFrame
    '''
    scaler = joblib.load(scaler) 
    return scaler.transform(data)

def balancer(data, features_selected, target):
    '''
    Balance data with SMOTE
    :params: data
    : X, y
    '''
    smote = SMOTE(random_state=9)
    X, y = smote.fit_resample(data[features_selected], data[target])
    return X,y

def data_splitter(X, y):
    '''
    Split data in train and test samples
    :params: X, y
    :return: X_train, X_test, y_train, y_test
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test



# Training function

def model_trainer(X_train, y_train, output_path):
    '''
    Train the model and store it
    :params: X_train, y_train, output_path
    :return: None
    '''
    # initialise the model
    rfor = RandomForestClassifier(max_depth=25, 
                                  min_samples_split=5, 
                                  n_estimators=300,
                                  random_state=8)
       
    # train the model
    rfor.fit(X_train, y_train)
    
    # save the model
    initial_type = [('features_input', FloatTensorType([1, X_train.shape[1]]))]
    onnx = convert_sklearn(rfor, name='rf_champion', initial_types=initial_type)
    with open(output_path, "wb") as f:
        f.write(onnx.SerializeToString())
        f.close()
    return None

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




    
