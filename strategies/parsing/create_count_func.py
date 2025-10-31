

def create_count_func(fraction_func, period, weight_counts):
    def count_func(i):
        if i != 0 and i % period == 0:
            return [int(fraction_func(i) * wc) for wc in weight_counts]
        return None
    return count_func