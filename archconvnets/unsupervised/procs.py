# procs - Run Python functions in parallel processes

# Copyright (c) 2007, Lenny Domnitser
# All rights reserved.
#
# Redistribution and use of this software in source and binary forms,
# with or without modification, are permitted provided that the
# following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''Run Python functions in parallel processes

Parallel child processes cannot affect in-memory state of the parent
or sibling processes, so procs is best suited for parallelizing
functional-style programs.

Procs uses pickle to share objects between processes.


Example usage:

    from procs import *

    # Some dummy functions we'll use below. Realistically, procs has
    # too much overhead to be useful for such small functions.
    def f(x, y):
        return 5 * x + 3 * y
    def g(n):
        return n + 4

    # Using call to run 4 processes in parallel:
    proc_specs = [          # Process specifications equivalent to:
        proc(f, 2, 7),      # f(2, 7)
        proc(f, x=-3, y=8), # f(x=-3, y=8)
        proc(g, 38),        # g(38)
        proc(g, n=95)       # g(n=95)
        ]
    vals = call(proc_specs) # Now the functions are actually called
    print list(vals)        # [31, 9, 42, 99]

    # Using pmap as a process-parallelized version of Python's map:
    vals = pmap(f, [12, 75, -2, 9], [5, -6, 0, 23])
    print list(vals)        # [75, 357, -10, 114]

    # Careful. Objects are pickled and copied:
    x = object()
    y = (lambda obj: obj)(x)
    print x is y            # True
    z = call([proc((lambda obj: obj), x)]).next()
    print x is z            # False

Exception handling:

    # When an error occurs in a worker process, a traceback is
    # printed, but all processes finish and yield a value. Failed
    # processes yield an instance of the procs.Failed exception, with
    # the original exception as its arguments.

    def double_odd(x):
        if x % 2 == 1:
            return x + x
        else:
            raise ValueError('not odd', x)

    print list(procs.pmap(double_odd, [1, 3, 6, 7]))
    # [2, 6, Failed(ValueError('not odd', 6),), 14]
'''

__version__ = 'dev'
__author__ = 'Lenny Domnitser <http://domnit.org/>'


__all__ = 'proc', 'call', 'pmap', 'Failed'


import itertools
import os
import sys
import traceback
try:
    import cPickle as pickle
except ImportError:
    import pickle


class Failed(Exception):
    pass


def proc(callback, *args, **kwargs):
    '''Builds a process specification

    For example, to perform the call foo('bar', 97, spam='eggs') in a
    new process, the process specification is
    proc(foo, 'bar', 97, spam='eggs').'''

    return callback, args, kwargs


def call(procs):
    '''Given an iterable of process specifications (see proc), calls
    each in a new process. Yields the return values of the callbacks
    in order, as they are ready. '''
    # The real code is in _call. This pops a bogus value so that the
    # processes are started without having to access the return values
    gen = _call(procs)
    gen.next()
    return gen


def _call(procs):
    outr, outw = os.pipe()
    for callback, args, kwargs in procs:
        inr, inw = os.pipe()
        pickle.dump((args, kwargs), os.fdopen(inw, 'w'))
        if os.fork() == 0:
            try:
                args, kwargs = pickle.load(os.fdopen(inr))
                try:
                    val = callback(*args, **kwargs)
                except:
                    val = Failed(sys.exc_info()[1])
                    traceback.print_exc()
                finally:
                    pickle.dump(val, os.fdopen(outw, 'w'))
            except:
                traceback.print_exc()
            finally:
                os._exit(0)

    yield # bogus. See call

    os.close(outw)
    outputreader = os.fdopen(outr)
    while True:
        try:
            val = pickle.load(outputreader)
            yield val
        except EOFError:
            break
    outputreader.close()


_exhausted = itertools.repeat(None)

def _map_proc(function, sequences):
    sequences = [iter(sequence) for sequence in sequences]
    unfinished = len(sequences)
    while True:
        args = []
        for i in xrange(len(sequences)):
            sequence = sequences[i]
            try:
                args.append(sequence.next())
            except StopIteration:
                sequences[i] = _exhausted
                unfinished -= 1
        if not unfinished:
            break
        yield proc(function, *args)
        


def pmap(function, *sequences):
    '''Like Python's built-in map, but runs in parallel processes.'''

    return call(_map_proc(function, sequences))

