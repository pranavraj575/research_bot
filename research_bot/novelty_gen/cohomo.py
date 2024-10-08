import os, numpy as np
from ripser import ripser
from persim import plot_diagrams
from matplotlib import pyplot as plt
import itertools
from research_bot.novelty_gen.novelty import Novelty


def drawLineColored(X, C):
    for i in range(X.shape[0] - 1):
        plt.plot(X[i:i + 2, 0], X[i:i + 2, 1], c=C[i, :], linewidth=2)


def plotCocycle2D(D, X, cocycle=None, thresh=1., colors=None):
    """
    Given a 2D point cloud X, display a cocycle projected
    onto edges under a given threshold "thresh"
    """
    # Plot all edges under the threshold
    N = X.shape[0]
    t = np.linspace(0, 1, 10)
    c = plt.get_cmap('Greys')
    C = c(np.array(np.round(np.linspace(0, 255, len(t))), dtype=np.int32))
    C = C[:, 0:3]

    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:
                Y = np.zeros((len(t), 2))
                Y[:, 0] = X[i, 0] + t*(X[j, 0] - X[i, 0])
                Y[:, 1] = X[i, 1] + t*(X[j, 1] - X[i, 1])
                drawLineColored(Y, C)
    # Plot cocycle projected to edges under the chosen threshold
    if cocycle is not None:
        for k in range(cocycle.shape[0]):

            [i, j, val] = cocycle[k, :]
            [i, j] = [min(i, j), max(i, j)]
            if D[i, j] <= thresh:
                a = 0.5*(X[i, :] + X[j, :])
                plt.text(a[0], a[1], '%g'%val, color='b')
                Y = np.zeros((len(t), 2))
                Y[:, 0] = X[i, 0] + t*(X[j, 0] - X[i, 0])
                Y[:, 1] = X[i, 1] + t*(X[j, 1] - X[i, 1])
            # drawLineColored(Y, C)
            # if D[i, j] <= thresh:
            #    [i, j] = [min(i, j), max(i, j)]
            #    a = 0.5*(X[i, :] + X[j, :])
            #    plt.text(a[0], a[1], '%g'%val, color='b')
    if colors is not None:
        for i, (x, y) in enumerate(X):
            color = colors[i%len(colors)]
            plt.scatter(x, y, color=color, s=80)
    # Plot vertex labels
    # return
    for i in range(N):
        plt.text(X[i, 0], X[i, 1], '%i'%i, color='r')
    plt.axis('equal')


def get_edge_list(D, X, thresh):
    # get all edges under the threshold
    out = []
    N = X.shape[0]

    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh and i != j:
                if {i, j} not in out:
                    out.append({i, j})
    return out


def facets(edges, vertices=None, current_set=None):
    if vertices is None:
        vertices = set()
        for (u, v) in edges:
            vertices.add(u)
            vertices.add(v)
    if current_set is None:
        current_set = set()

    def valid_addition(v):
        # checks if v can extend the facet, only happens if v is connected to all of the current facets
        for u in current_set:
            if {u, v} not in edges:
                return False
        return True

    out = []
    for v in vertices:
        if valid_addition(v):
            for facet in facets(edges=edges, vertices=vertices, current_set=current_set.union({v})):
                if facet not in out:
                    out.append(facet)
    if not out:
        out = [current_set]
    return out


def get_representative(data, dim):
    do_cocycles = False
    rip = ripser(data, maxdim=dim, do_cocycles=do_cocycles)
    if len(rip['dgms'][dim]) == 0:
        return None
    removed = []
    removable = True
    while removable:
        removable = False
        for i in range(len(data)):
            if not i in removed:
                test_removed = removed + [i]
                test_data = data[[j for j in range(len(data)) if j not in test_removed]]
                test_rip = ripser(test_data, maxdim=dim, do_cocycles=do_cocycles)
                if len(test_rip['dgms'][dim]) > 0:
                    removable = True
                    removed = test_removed
    return [j for j in range(len(data)) if j not in removed]


def get_event_simplices(event, dperm2all, degree=1, tol=1e-8):
    """
    gets simplices that form during an event (i.e. the birth or death of cohomology)
    Args:
        event: scalar, the time that the event happens
        dperm2all: distance matrix, returned by novelty_gen
        degree: degree of simplex we should return (should usually be 1)
            i.e. if we are looking for a line that caused the death of 0-cohomology, degree would be 1
            k-simplices cause death of (k-1)-cohomology and birth of k-cohomology
        tol: usually our threshold is the edge closest to the event, if tol is nonzero, we add tol to threshold
    returns simplices that have just formed at specified event with the specified degree
        since we are looking at rips complex, we inspect edges that form, and if degree is more than 1,
            we look for simplices that include the edge
    """
    diffs = np.abs(dperm2all - event)
    diff_thresh = np.min(diffs) + tol

    for i, j in zip(*np.where(diffs <= diff_thresh)):
        if i < j:  # remove duplicate edges
            if degree == 1:
                yield i, j
            else:
                # this is distance where edge (i,j) formed
                temp_event = dperm2all[i, j]

                # check all possible choices of remaining vertices where the total
                # (i,j,rest) is the number of points in a simplex of specified degree
                stuff = [k for k in range(len(dperm2all)) if k not in (i, j)]
                for rest in itertools.combinations(stuff, degree + 1 - 2):
                    arr = dperm2all[(i, j),][:, rest]
                    if np.all(arr <= temp_event + tol):
                        # then when edge (i,j) formed, the simplex (i,j, rest) also formed
                        yield (i, j) + rest


def event_catcher(rng, diagrams, reverse=True):
    events = []
    a, b = rng
    for degree, dgm in enumerate(diagrams):
        stuff = np.where(np.logical_and(a < dgm, dgm < b))
        for row, bd in zip(*stuff):
            events.append((dgm[row, bd], degree, bd == 0))
    events.sort(reverse=reverse)
    return events


def default_radius_bounds(dataset,
                          min_radius=None,
                          max_radius=None,
                          ):
    if min_radius is None:
        # then do double average distance to nearest neighbor
        dperm2all = np.linalg.norm(np.expand_dims(dataset, 0) - np.expand_dims(dataset, 1), axis=-1)
        temp = dperm2all + np.diag(np.nan*np.ones(len(dperm2all)))
        min_radius = 1.5*np.mean(np.nanmin(temp, axis=1))
    if max_radius is None:
        # then do double min radius
        max_radius = 2*min_radius
    return min_radius, max_radius


def phase_transitions(dataset,
                      min_radius,
                      max_radius,
                      max_cohomo=4,
                      check_simplices=True,
                      ):
    """
    Args:
        check_simplices: Whether to return the simplices added or just the lines
    """

    result = ripser(dataset, do_cocycles=False, maxdim=max_cohomo)
    dperm2all = result['dperm2all']
    diagrams = result['dgms']
    for event, degree, birth in event_catcher(rng=(min_radius, max_radius),
                                              diagrams=diagrams,
                                              reverse=False,
                                              ):
        if not birth:
            # death of cohomology of degree k implies appearance of a k+1 simplex
            degree += 1
        if not check_simplices:
            degree = 1
        for simplex in get_event_simplices(event=event, dperm2all=dperm2all, degree=degree):
            yield event, simplex


def barycentric_additions(dataset,
                          min_radius=None,
                          max_radius=None,
                          max_cohomo=4,
                          check_simplices=True,
                          ):
    if min_radius is None or max_radius is None:
        min_radius, max_radius = default_radius_bounds(dataset=dataset,
                                                       min_radius=min_radius,
                                                       max_radius=max_radius,
                                                       )
    for event, simplex in phase_transitions(dataset=dataset,
                                            min_radius=min_radius,
                                            max_radius=max_radius,
                                            max_cohomo=max_cohomo,
                                            check_simplices=check_simplices,
                                            ):
        barycenter = np.mean(dataset[simplex,], axis=0)
        yield barycenter


def stitch_together(dataset,
                    min_radius=None,
                    max_radius=None,
                    max_cohomo=4,
                    check_simplices=False,
                    depth=float('inf'),
                    ):
    if min_radius is None or max_radius is None:
        min_radius, max_radius = default_radius_bounds(dataset=dataset,
                                                       min_radius=min_radius,
                                                       max_radius=max_radius,
                                                       )
    # stitches dataset together until no cohomology is born/dies on specified range
    additions = []
    for event, simplex in phase_transitions(dataset=dataset,
                                            min_radius=min_radius,
                                            max_radius=max_radius,
                                            max_cohomo=max_cohomo,
                                            check_simplices=check_simplices,
                                            ):
        barycenter = np.mean(dataset[simplex,], axis=0)
        additions.append(barycenter)
    if not additions:
        return
    for b in additions:
        yield b
    if depth <= 1:
        return
    dataset = np.concatenate((dataset, np.stack(additions, axis=0)), axis=0)
    for thing in stitch_together(dataset=dataset,
                                 min_radius=min_radius,
                                 max_radius=max_radius,
                                 max_cohomo=max_cohomo,
                                 check_simplices=check_simplices,
                                 depth=depth - 1,
                                 ):
        yield thing


class HomologyNovelty(Novelty):
    def _generate_novelties(self, n, dataset):
        pass


if __name__ == '__main__':
    # resolution
    # lower radius is 'research_bot', basically saying ignore points in smaller resolution than this
    # upper radius is 'coherence', basically saying dont get too crazy now
    min_radius = .5
    max_radius = np.inf
    N = 8

    np.random.seed(2)
    t = np.linspace(0, 2*np.pi, N + 1)[0:N]
    x = np.array([np.cos(t), np.sin(t)]).T
    x += np.random.randn(x.shape[0], 2)*0.25
    result = ripser(x, do_cocycles=True)
    dperm2all = result['dperm2all']
    print(dperm2all)
    plt.scatter(x[:, 0], x[:, 1])
    plt.axis('equal')
    plt.show()
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']
    murderers = []
    mmm = []

    for event, degree, birth in event_catcher(rng=(min_radius, max_radius), diagrams=diagrams, reverse=False):
        if not birth:
            # death of cohomology of degree k implies appearance of a k+1 simplex
            degree += 1

        print('birth' if birth else 'death', 'event at', event, 'degree', degree)
        for simplex in get_event_simplices(event=event, dperm2all=dperm2all, degree=degree):
            barycenter = np.mean(x[simplex,], axis=0)
            print('\tsimplex', simplex, 'center', barycenter)
            if birth:
                mmm.append(barycenter)
            else:
                murderers.append(barycenter)

    pts = []
    for event, simplex in phase_transitions(dataset=x, min_radius=min_radius, max_radius=max_radius,
                                            max_cohomo=1, ):
        barycenter = np.mean(x[simplex,], axis=0)
        pts.append((event, barycenter))
    dgm1 = diagrams[1]
    idx = np.argmax(dgm1[:, 1] - dgm1[:, 0])  # index of most persistent 1 cohomology
    plot_diagrams(diagrams, show=False)
    plt.show()

    cocycle = cocycles[1][idx]

    birth_thresh = dgm1[idx, 0]
    for edge in get_event_simplices(birth_thresh, dperm2all=dperm2all):
        print(edge)

    birth_thresh = birth_thresh + 0.01
    plotCocycle2D(D, x, cocycle, birth_thresh)
    plt.scatter([t[0] for t in mmm], [t[1] for t in mmm], zorder=69)
    plt.title("post birth at %g"%birth_thresh)
    plt.show()

    death_thresh = dgm1[idx, 1]
    for edge in get_event_simplices(death_thresh, dperm2all=dperm2all):
        print(edge)
    death_thresh = death_thresh - .01
    plotCocycle2D(D, x, cocycle, death_thresh)
    plt.scatter([t[0] for t in murderers], [t[1] for t in murderers], zorder=69)
    plt.title("pre death at %g"%death_thresh)
    plt.show()

    plt.scatter(x[:, 0], x[:, 1], color='black', label='dataset')
    plt.scatter([t[0] for t in mmm], [t[1] for t in mmm], label='birth')
    plt.scatter([t[0] for t in murderers], [t[1] for t in murderers], label='death')
    plt.legend()
    plt.axis('equal')
    plt.show()

    plt.scatter(x[:, 0], x[:, 1], color='black', label='dataset')
    event_min = min([ev for ev, _ in pts])
    event_max = max([ev for ev, _ in pts])
    for event, point in pts:
        if event_max > event_min:
            t = (event - event_min)/(event_max - event_min)
            rgb = []
            for item in [t, 0, 1 - t]:
                item = hex(int(item*255))[2:]
                while len(item) < 2:
                    item = '0' + item
                rgb.append(item)
            color = '#' + ''.join(rgb)
        else:
            color = 'purple'
        plt.scatter(point[0], point[1], color=color)
    plt.legend()
    plt.axis('equal')
    plt.show()
