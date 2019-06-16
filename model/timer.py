import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


if __name__ == '__main__':

    start = time.time()
    print_loss_avg = 0.5
    iter = 1
    n_iters = 10000
    print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                 iter, iter / n_iters * 100, print_loss_avg))