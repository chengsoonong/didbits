"""Utility script to be used to cleanup the notebooks before git commit

This a mix from @minrk's various gists.

Copied from Olivier Griesel's parallel ipython tutorial:
https://github.com/ogrisel/parallel_ml_tutorial

Added: https://gist.github.com/minrk/3836889

"""

import time
import sys
import os
import io
import argparse
try:
    from queue import Empty
except:
    # Python 2 backward compat
    from Queue import Empty

import nbformat
from jupyter_client import KernelManager
from ipyparallel import Client

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


assert KernelManager  # to silence pyflakes


def remove_outputs(nb):
    """Remove the code outputs from a notebook"""
    num_outputs = 0
    for cell in nb.cells:
        if cell.cell_type == 'code':
            num_outputs += 1
            cell.outputs = []
            if 'execution_count' in cell:
                cell['execution_count'] = None
    print('removed %d code outputs' % num_outputs)

def remove_solution_code(nb):
    """Remove code cells that start with # Solution"""
    scrubbed = 0
    cells = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        cells += 1
        # scrub cells marked with initial '# Solution' comment
        # any other marker will do, or it could be unconditional
        if cell.source.startswith('# Solution'):
            cell.source = u'# Solution goes here'
            scrubbed += 1
            cell.outputs = []
    #print('scrubbed %i/%i code cells from notebook %s' % (scrubbed, cells, nb.metadata.name))
    print('scrubbed %i/%i code cells from notebook' % (scrubbed, cells))

def remove_solution_text(nb):
    """Remove markdown cells that start with ### Solution"""
    scrubbed = 0
    cells = 0
    for cell in nb.cells:
        if cell.cell_type != 'markdown':
            continue
        cells += 1
        # scrub cells marked with initial '### Solution' comment on a line by itself
        if cell.source.startswith(u'### Solution'):
            scrubbed += 1
            cell.source = u'### Solution description'
    print('scrubbed %i/%i markdown cells from notebook' % (scrubbed, cells))

def run_cell(kernel_client, cell, timeout=300):
    if not hasattr(cell, 'source'):
        return [], False
    kernel_client.execute(cell.source)
    # wait for finish, maximum 5min by default
    reply = kernel_client.get_shell_msg(timeout=timeout)['content']
    if reply['status'] == 'error':
        failed = True
        print("\nFAILURE:")
        print(cell.source)
        print('-----')
        print("raised:")
        print('\n'.join(reply['traceback']))
    else:
        failed = False

    # Collect the outputs of the cell execution
    outs = []
    while True:
        try:
            msg = kernel_client.get_iopub_msg(timeout=0.2)
        except Empty:
            break
        msg_type = msg['msg_type']
        if msg_type in ('status', 'execute_input'):
            continue
        elif msg_type == 'clear_output':
            outs = []
            continue

        content = msg['content']
        out = nbformat.NotebookNode(output_type=msg_type)
        if msg_type == 'stream':
            out.name = content['name']
            out.text = content['text']
        elif msg_type in ('display_data', 'execute_result'):
            for mime, data in content['data'].items():
                attr = mime.split('/')[-1].lower()
                # this gets most right, but fix svg+html, plain
                attr = attr.replace('+xml', '').replace('plain', 'text')
                setattr(out, attr, data)
            if msg_type == 'execute_result':
                out.execution_count = content['execution_count']
        elif msg_type == 'error':
            out.ename = content['ename']
            out.evalue = content['evalue']
            out.traceback = content['traceback']
        elif msg_type == 'execute_input':
            print(content)
        else:
            print("unhandled iopub msg: %s" % msg_type)
        outs.append(out)

    # Special handling of ipcluster restarts
    if '!ipcluster stop' in cell.source:
        # wait some time for cluster commands to complete
        for i in range(10):
            try:
                if len(Client()) == 0:
                    break
            except FileNotFoundError:
                pass
            sys.stdout.write("@"); sys.stdout.flush()
            time.sleep(5)
    if '!ipcluster start' in cell.source:
        # wait some time for cluster commands to complete
        for i in range(10):
            try:
                if len(Client()) > 0:
                    break
            except FileNotFoundError:
                pass
            sys.stdout.write("#"); sys.stdout.flush()
            time.sleep(5)
    return outs, failed


def run_notebook(nb):
    km = KernelManager()
    km.start_kernel(stderr=open(os.devnull, 'w'))
    kc = km.client()
    kc.start_channels()

    # simple ping:
    kc.execute("pass")
    kc.get_shell_msg()

    cells = 0
    failures = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue

        outputs, failed = run_cell(kc, cell)
        cell.outputs = outputs
        cells += 1
        cell['execution_count'] = cells
        failures += failed
        sys.stdout.write('.')
        sys.stdout.flush()

    print()
    #print("ran notebook %s" % nb.metadata.name)
    print("    ran %3i cells" % cells)
    if failures:
        print("    %3i cells raised exceptions" % failures)
    kc.stop_channels()
    km.shutdown_kernel()
    del km


def process_notebook_file(fname, action='clean', output_fname=None):
    print("Performing '{}' on: {}".format(action, fname))
    orig_wd = os.getcwd()
    with io.open(fname, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    if action == 'check':
        os.chdir(os.path.dirname(fname))
        run_notebook(nb)
        remove_outputs(nb)
    elif action == 'render':
        os.chdir(os.path.dirname(fname))
        run_notebook(nb)
    elif action == 'worksheet':
        os.chdir(os.path.dirname(fname))
        run_notebook(nb)
        remove_outputs(nb)
        remove_solution_text(nb)
        remove_solution_code(nb)
    else:
        # Clean by default
        remove_outputs(nb)

    os.chdir(orig_wd)
    if output_fname is None:
        output_fname = fname
    with io.open(output_fname, 'w') as f:
        nb = nbformat.write(nb, f)

def take_action(action, targets):
    rendered_folder = os.path.join(os.path.dirname(__file__), '../../tutorial')
    if not os.path.exists(rendered_folder):
        os.makedirs(rendered_folder)
    if not targets:
        targets = [os.path.join(os.path.dirname(__file__), 'notebooks')]

    for target in targets:
        if os.path.isdir(target):
            fnames = [os.path.abspath(os.path.join(target, f))
                      for f in os.listdir(target)
                      if f.endswith('.ipynb')]
        else:
            fnames = [target]
        for fname in fnames:
            if action == 'render':
                new_name = os.path.splitext(os.path.basename(fname))[0] + '-sol.ipynb'
                output_fname = os.path.join(rendered_folder, new_name)
            elif action == 'worksheet':
                new_name = os.path.splitext(os.path.basename(fname))[0] + '-work.ipynb'
                output_fname = os.path.join(rendered_folder, new_name)
            else:
                output_fname = fname
            process_notebook_file(fname, action=action,
                                  output_fname=output_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', help='List of directories to apply action to',
                        action='append')
    parser.add_argument('-c', '--clean', help='Clean up generated files',
                        action='store_true')
    parser.add_argument('-d', '--debug', help='Check that notebooks are ok',
                        action='store_true')
    parser.add_argument('-r', '--render', help='Render notebooks to outpath',
                        action='store_true')
    parser.add_argument('-s', '--solution', help='Render notebook with solutions',
                        action='store_true')
    parser.add_argument('-o', '--outpath', help='Output path for rendered notebooks',
                        )
    args = parser.parse_args()

    targets = [t for t in args.targets]
    if args.render:
        if args.solution:
            take_action('render', targets)
        else:
            take_action('worksheet', targets)
    if args.debug:
        take_action('check', targets)
    if args.clean:
        take_action('clean', targets)
