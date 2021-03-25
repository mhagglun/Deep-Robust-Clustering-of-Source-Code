#!/usr/bin/env python
import os
import re
import ast
import sys
import glob
import json
import shutil
import random
import logging
import jsonlines
import astunparse
import numpy as np

from tqdm import tqdm
from filelock import FileLock
from argparse import ArgumentParser
from collections import namedtuple, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from pyminifier import pyminify
from pytransformation import String_Transformer
from pytransformation import Transformations

class Options(object):
    prepend = None
    replacement_length = 1
    use_nonlatin = False
    obf_builtins = False
    obf_import_methods = True
    obf_variables = True
    obf_functions = True
    obf_classes = True
    obfuscate = False
    tabs = False
    nominify = False
    destdir = "./minified"
    outfile = None


options = Options()
transformations = [
    Transformations.MAKE_FOR_LOOPS_WHILE,
    Transformations.MOVE_DECLS_TO_TOP_OF_SCOPE,
    Transformations.NAME_ALL_FUNCTION_APPLICATIONS,
]


class NodeVisitor(ast.NodeVisitor):
    def __init__(self, fname):
        self.fname = fname
        self.filters = ['train', 'save', 'process', 'forward', 'predict']

    def visit_FunctionDef(self, node):
        method_name = node.name
        if any([True for s in self.filters if method_name.find(s) != -1]): # check if name matches filter
            if not any([True for idx, child in enumerate(ast.walk(node)) if idx != 0 and isinstance(child, ast.FunctionDef)]): # check that there are no nested methods
                source = astunparse.unparse(node)
                try:
                    transformed_source = pyminify(options, self.fname, source) # Rename variables, remove comments

                    transformer = String_Transformer(transformations)
                    transformed_source = transformer.transform(transformed_source) # Convert loops, move declarations and name function calls
                    if len(transformed_source) == 0:
                        raise Exception                        
                    ast.parse(transformed_source) # Parse the transformed code to verify that it's functionally intact
                except Exception as e:
                    return

                target_path = f"{'-'.join(self.fname.split('/')[-2:])}"
                with open(f"./data/class_samples_large/{target_path}", 'a') as f1:
                    f1.write(source + "\n" + transformed_source)

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)

class Extractor():
    """Extract methods from directory"""
    def __init__(self, args):
        self.args = args

    def extractor(self, fname):
        with open(fname, 'r') as input_file:
            try:
                root = ast.parse(input_file.read(), fname)
            except Exception as e:
                if self.args.verbose:
                    print(
                        f"Skipping problematic file {e}", fname, file=sys.stderr)
                return
        nodeVisitor = NodeVisitor(fname)
        for node in ast.iter_child_nodes(root):
            nodeVisitor.visit(node)

    def extract(self):
        projects = glob.glob(self.args.directory+'/**/*.py', recursive=True)
        with tqdm(total=len(projects), unit=' Files', desc='Extracting methods from files') as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                tasks = {executor.submit(
                    self.extractor, file): file for file in projects}

                for task in as_completed(tasks):
                    pbar.update(1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", type=str,
                        help="The directory of the projects to extract libraries from", required=True)
    parser.add_argument("-v", "--verbose", dest="verbose", type=bool, default=False,
                        help="Increase verbosity of output", required=False)

    args = parser.parse_args()
    extractor = Extractor(args)
    extractor.extract()

    extracted_files = glob.glob('./data/class_samples_large/**/*.py', recursive=True)
    random.shuffle(extracted_files)

    # Partition files into an 80/10/10 split
    train, validate, test = np.split(extracted_files, [int(.8*len(extracted_files)), int(.9*len(extracted_files))])

    for ds_name, ds in zip(['train', 'val', 'test'], [train, validate, test]):
        target_dir = f'./data/class_samples_large/{ds_name}'
        os.mkdir(target_dir)
        for f in ds:
            shutil.move(f, target_dir)
    print("Done.")
