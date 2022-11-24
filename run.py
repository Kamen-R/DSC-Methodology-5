from test_model import train_model

def main(target):

    if targets == "test":
        with open('test/testdata.csv') as file:
            test_data = pd.read_csv(file)
        
        model = train_model(test_data)
        
    return

def train_model(data):
    y = pd.to_datetime(data["summary_date"]).apply(datetime.date.weekday).apply(lambda x: "Weekday" if x < 5 else "Weekend")
    X = data
    X = X.merge(pd.DataFrame(test_patient["hypnogram_5min"].apply(list).to_list()), how="outer", left_index=True, right_index=True)
    X = X.drop(columns=["summary_date", "type", "timestamp", "hypnogram_5min"])
    
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X, y)
    
    return model

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)