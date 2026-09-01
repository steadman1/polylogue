from collections.abc import Generator


# written by Gemini
def generator_check_last[T](
    generator: Generator[T, None, None],
) -> Generator[(bool, T), None, None]:
    iterable = iter(generator)

    try:
        # Pull the first item to look ahead
        current_item = next(iterable)
    except StopIteration:
        # The generator was empty from the start
        return

    for next_item in iterable:
        # If the loop continues, current_item is NOT the last item
        yield False, current_item
        current_item = next_item

    # The loop finished, meaning there are no more items left
    yield True, current_item
