import tenseal as ts
import os
import pickle
import numpy as np

from encryption.paillier import PaillierCipher

def context():
    """
    This function is used to create the context of the homomorphic encryption:
    it is used to create the keys and the parameters of the encryption scheme (CKKS).

    :return: the context of the homomorphic encryption
    """
    cont = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        # This means that the coefficient modulus will contain 4 primes of 60 bits, 40 bits, 40 bits, and 60 bits.
        coeff_mod_bit_sizes=[60, 40, 40, 60]
        # coeff_mod_bit_sizes=[40, 21, 21, 40]
    )

    cont.generate_galois_keys()  # You can create the Galois keys by calling generate_galois_keys
    # cont.generate_relin_keys()
    cont.global_scale = 2 ** 40  # global_scale: the scaling factor, here set to 2**40 (same that pow(2, 40))
    return cont

def read_query(file_path):
    """
    This function is used to read a pickle file.

    :param file_path: the path of the file to read
    :return: the query and the context
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            """
            # pickle.load(f)  # load to read file object

            file_str = f.read()
            client_query1 = pickle.loads(file_str)  # loads to read str class
            """
            query_str = pickle.load(file)

        contexte = query_str["context"]  # ts.context_from(query["contexte"])
        del query_str["context"]
        return query_str, contexte

    else:
        print("The file doesn't exist")

def write_query(file_path, client_query):
    """
    This function is used to write a pickle file.

    :param file_path: the path of the file to write

    :param client_query: the query to write
    """
    with open(file_path, 'wb') as file:  # 'ab' to add existing file
        encode_str = pickle.dumps(client_query)
        file.write(encode_str)

def combo_keys(client_path="secret.pkl", server_path="server_key.pkl"):
    """
    To create the public/private keys combination
    args:
        client_path: path to save the secret key (str)
        server_path: path to save the server public key (str)
    """
    context_client = context()
    write_query(f'context/{client_path}', {"context": context_client.serialize(save_secret_key=True)})
    write_query(f'context/{server_path}', {"context": context_client.serialize()})

    _, context_client = read_query(f'context/{client_path}')
    _, context_server = read_query(f'context/{server_path}')

    context_client = ts.context_from(context_client)
    context_server = ts.context_from(context_server)
    print("Is the client context private?", ("Yes" if context_client.is_private() else "No"))
    print("Is the server context private?", ("Yes" if context_server.is_private() else "No"))


def Paillier_keys():
    paillier = PaillierCipher()
    paillier.generate_key(n_length=2048)
    
    with open("context/paillier.pkl", "wb") as file:
        pickle.dump(paillier, file)

def main():
    combo_keys()
    Paillier_keys()

if __name__ == "__main__":
    main()