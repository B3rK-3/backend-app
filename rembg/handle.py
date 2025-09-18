from rembg.bg import remove
import onnxruntime as ort
import base64
from rembg.u2net import U2netSession
import os



def handle(b64In: bytes):
    """
    takes in a bytes object
    
    returns bytes object
    """
    out = start("u2net", b64In)
    return out


def start(model: str, input: bytes) -> bytes:
    """
    Click command line interface function to process an input file based on the provided options.

    This function is the entry point for the CLI program. It reads an input file, applies image processing operations based on the provided options, and writes the output to a file.

    Parameters:
        model (str): The name of the model to use for image processing.
        extras (str): Additional options in JSON format.
        input: The input file to process.
        output: The output file to write the processed image to.
        **kwargs: Additional keyword arguments corresponding to the command line options.

    Returns:
        A bytes object
    """
    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        threads = int(os.environ["OMP_NUM_THREADS"])
        sess_opts.inter_op_num_threads = threads
        sess_opts.intra_op_num_threads = threads

    return remove(input, session=U2netSession(model, sess_opts))


