import os

def write_train_logs(rnd, cid, loss, acc, model_size, train_time, cyfer_time, decyfer_time, dataset, solution):
    filename = f'../logs/{dataset}/{solution}/train.csv' 
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'a') as file:
        if os.path.getsize(filename) == 0:
            file.write("round, cid, loss, accuracy, model_size, train_time, cyfer_time, decyfer_time\n")
        file.write(f"{rnd}, {cid}, {loss}, {acc}, {model_size}, {train_time}, {cyfer_time}, {decyfer_time}\n")


def write_evaluate_logs(rnd, cid, loss, acc, decyfer_time, dataset, solution):
    filename = f'logs/{dataset}/{solution}/evaluate.csv' 
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'a') as file:
        if os.path.getsize(filename) == 0:
            file.write("round, cid, loss, accuracy, decyfer_time\n")
        file.write(f"{rnd}, {cid}, {loss}, {acc}, {decyfer_time}\n")