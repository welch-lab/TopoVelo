import inspect
import numpy as np
import cProfile
import pstats
from memory_profiler import memory_usage


def get_arguments(func):
    """Get the arguments of a function
    """
    # Get the signature of the __init__ function
    func_signature = inspect.signature(func)
    func_params = func_signature.parameters

    # Extract the argument names and optional argument names
    argument_names = []
    optional_argument_names = []
    optional_argument_values = []
    for param_name, param in func_params.items():
        if param_name == "self" or param_name == "kwargs":
            continue
        if param.default != inspect.Parameter.empty:
            optional_argument_names.append(param_name)
            optional_argument_values.append(param.default)
        else:
            argument_names.append(param_name)
    return argument_names, optional_argument_names, optional_argument_values


class ModelProfiler():
    """Profiler for a model
    """
    def __init__(self, model_class, *args, **kwargs):
        """Profile the initialization of the model

        Args:
            model_class: the class of the model
            *args: the arguments of the model
        """
        self.model_class = model_class
        # Get the arguments of the model
        arg_names, opt_arg_names, opt_arg_values = get_arguments(model_class.__init__)
        if len(args) > len(arg_names):
            raise ValueError("Too many arguments")
        # Fill in the optional arguments
        for i, kw in enumerate(opt_arg_names):
            if kw not in kwargs:
                kwargs[kw] = opt_arg_values[i]
        self.args = args
        self.kwargs = kwargs

        self.prof_train = None
        self.train_stats = {}
        self.prof_infer = None
        self.infer_stats = {}
        self.is_cuda = False
        
        self.elapsed_time_stats = None
        self.max_memory = None

    def profile_cpu_memory(self, *args, **kwargs):
        """Profile the CPU memory of the model
        """
        self.model = self.model_class(*self.args, **self.kwargs)
        mem = memory_usage(
            (self.model.train, args, kwargs),
            max_usage=True
        )
        self.max_memory = mem / 1024 / 1024 # in MB
    
    def profile_elapsed_time(self, *args, **kwargs):
        """Get the elapsed time of the entire pipeline from model creation to training
        Since CPU and GPU programs are asynchrnous, we only measure the actual clock time
        until the program finishes.

        Returns:
            float: the elapsed time in minutes
        """
        # Get the arguments of the model
        arg_names, opt_arg_names, opt_arg_values = get_arguments(self.model_class.train)
        if len(args) > len(arg_names):
            raise ValueError("Too many arguments")
        # Fill in the optional arguments
        for i, kw in enumerate(opt_arg_names):
            if kw not in kwargs:
                kwargs[kw] = opt_arg_values[i]
        self.model = self.model_class(*self.args, **self.kwargs)
        # Run the model
        profiler = cProfile.Profile()
        profiler.enable()
        self.model.train(*args, **kwargs)
        profiler.disable()
        self.elapsed_time_stats = pstats.Stats(profiler).sort_stats('cumulative')
    
    @property
    def total_elapsed_time(self):
        """Get the elapsed time of the entire pipeline from model creation to training

        Returns:
            float: the elapsed time in minutes
        """
        return self.elapsed_time_stats.total_tt/60
    
    @property
    def elapsed_time_per_epoch(self):
        """Get the elapsed time per iteration

        Returns:
            float: the elapsed time in seconds
        """
        for key, val in self.elapsed_time_stats.stats.items():
            if key[2] == 'train_epoch' and 'topovelo' in key[0]:
                return val[3]/val[1]
        return np.nan
