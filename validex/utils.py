import functools
import hashlib
import os
import pickle
import time
from collections.abc import Callable


def async_cache_to_disk(expiration_time: int = 182 * 24 * 60 * 60) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a unique cache key based on function name and arguments
            args_kwargs_str = str(args) + str(kwargs)
            hash_object = hashlib.sha256(args_kwargs_str.encode())
            cache_key = f"{func.__name__}_{hash_object.hexdigest()}"
            cache_file = f"{cache_key}.pkl"

            # Check if cache file exists and is not expired
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        cached_time, cached_result = pickle.load(f)
                    if time.time() - cached_time <= expiration_time:
                        print(f"Cache hit for {func.__name__}")
                        return cached_result
                except (pickle.PickleError, EOFError, ValueError):
                    print(f"Error reading cache for {func.__name__}, ignoring cache")

            # If cache doesn't exist, is expired, or couldn't be read, call the original function
            print(f"Cache miss for {func.__name__}, calling function")
            result = await func(*args, **kwargs)

            # Save the result to cache
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump((time.time(), result), f)
                print(f"Cached result for {func.__name__}")
            except (pickle.PickleError, OSError):
                print(f"Error caching result for {func.__name__}")

            return result

        return wrapper

    return decorator
