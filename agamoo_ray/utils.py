import numpy as np


CEXT = False
try:
    import agamoo_ray.cutils as cutils
    CEXT = True
except Exception as e:
    print('C extension not available')
    print(e)
    CEXT = False


def pairwise_dominance(x):
    z = x[:, np.newaxis] >= x #org
    z = np.all(z, axis=2)
    z[range(z.shape[0]), range(z.shape[0])] = False
    #z[np.triu_indices(z.shape[0])] = False
    xx = np.any(z, axis=1)
    return np.logical_not(xx)


def get_not_dominated(populations_eval):
    if CEXT:
        mask = np.zeros(populations_eval.shape[0], dtype=np.int32)
        cutils.cget_not_dominated(populations_eval, mask)
        mask = mask.astype(bool)
    else:
        mask = pairwise_dominance(populations_eval)
    return mask


def pairwise_distance(x):
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def front_suppression(front_eval, front_max):
    if CEXT:
        mask = np.zeros(front_eval.shape[0], dtype=np.int32)
        cutils.cfront_suppression(front_eval, front_max, mask)
        mask = mask.astype(bool)
    else:
        n = front_eval.shape[0] - front_max
        ideal = np.argmin(front_eval, axis=0)
        front_eval_norm = front_eval + np.abs(np.min(front_eval, axis=0))+1.0
        front_eval_norm = front_eval_norm/np.max(front_eval_norm, axis=0)
        z = pairwise_distance(front_eval_norm)
        mask = np.ones(front_eval.shape[0], dtype=bool)
        t = np.tril(z) + np.triu(np.ones_like(z) * 1000000)
        arg = np.argsort(t, axis=None)
        indx_i, indx_j = np.unravel_index(arg, t.shape)
        while n > 0:
            ii = indx_i[0]
            mask[ii] = False
            tmp = indx_i[indx_i!=ii]
            indx_j = indx_j[indx_i!=ii]
            indx_i = tmp.copy()
            tmp = indx_j[indx_j!=ii]
            indx_i = indx_i[indx_j!=ii]
            indx_j = tmp.copy()
            n=n-1
        for i in ideal:
            mask[i] = True
    return mask


def assigning_gens(nvars, nobjs):
    while True:
        if nvars <= nobjs:
            #r = np.random.choice(range(nobjs), size=(nvars,), replace=False)
            #r2 = np.stack([r == i for i in range(nobjs)])
            r = np.random.choice(range(nvars), size=(nobjs,), replace=True)
            r2 = np.stack([r == i for i in range(nvars)])
            r2 = r2.T
            break
        else:
            r = np.random.randint(0, nobjs, size=(nvars,))
            r2 = np.stack([r == i for i in range(nobjs)])
            if nvars >= nobjs and not np.any(np.all(r2, axis=1)) and not np.any(np.all(np.logical_not(r2), axis=1)):
                break
    return r2


def adaptive_linear_assigning_gens(front, front_eval, nvars, nobjs):
    """
    Adaptacyjnie przypisuje zmienne do graczy na podstawie korelacji Pearsona
    między wartością zmiennej a wartością funkcji celu na obecnym froncie.
    """
    # Fallback: Jeśli front jest zbyt mały do analizy statystycznej,
    # używamy zwykłego losowego przypisania (fallback do oryginalnej funkcji)
    if len(front) < 10:
        return assigning_gens(nvars, nobjs)

    patterns = np.zeros((nobjs, nvars), dtype=bool)
    corr_matrix = np.zeros((nvars, nobjs))

    for i in range(nvars):
        var_data = front[:, i]
        if np.std(var_data) > 0:
            for j in range(nobjs):
                obj_data = front_eval[:, j]
                if np.std(obj_data) > 0:
                    # Zapisujemy bezwzględną wartość korelacji do macierzy
                    corr_matrix[i, j] = np.abs(np.corrcoef(var_data, obj_data)[0, 1])

    for i in range(nvars):
        corrs = corr_matrix[i, :]
        sum_corrs = np.sum(corrs)

        if sum_corrs > 0:
            probs = corrs / sum_corrs
        else:
            probs = np.ones(nobjs) / nobjs  # Brak korelacji -> równa szansa

        chosen_obj = np.random.choice(nobjs, p=probs)
        patterns[chosen_obj, i] = True

    empty_players = np.where(~np.any(patterns, axis=1))[0]

    for j in empty_players:
        # Pętla aktualizuje "bogatych graczy", bo po kradzieży ktoś mógł przestać być bogaty
        rich_players = np.where(np.sum(patterns, axis=1) > 1)[0]

        if len(rich_players) > 0:
            # === SCENARIUSZ A: KRADZIEŻ ===
            # Znajdujemy wszystkie zmienne, które można bezpiecznie ukraść
            stealable_vars = np.where(np.any(patterns[rich_players], axis=0))[0]

            # Wybieramy tę zmienną, która ma najwyższą korelację z naszym pustym graczem 'j'
            obj_corrs = corr_matrix[stealable_vars, j]
            if np.sum(obj_corrs) > 0:
                best_stealable_idx = np.argmax(obj_corrs)
            else:
                best_stealable_idx = np.random.choice(len(stealable_vars))

            var_to_steal = stealable_vars[best_stealable_idx]

            # Znajdujemy "ofiarę" (bogatego gracza, który ma ten konkretny gen)
            owners = np.where(patterns[:, var_to_steal])[0]
            rich_owners = np.intersect1d(owners, rich_players)
            victim = rich_owners[0]

            # Wykonujemy transfer (Zabieramy ofierze, dajemy pustemu)
            patterns[victim, var_to_steal] = False
            patterns[j, var_to_steal] = True

        else:
            # === SCENARIUSZ B: WSPÓŁDZIELENIE ===
            # Wszyscy gracze mają już po 1 genie. Nie możemy kraść.
            # Wybieramy absolutnie najlepszy gen z całej puli.
            obj_corrs = corr_matrix[:, j]
            if np.sum(obj_corrs) > 0:
                best_var = np.argmax(obj_corrs)
            else:
                best_var = np.random.choice(nvars)

            # Przyznajemy uprawnienia (BEZ odbierania ich obecnemu właścicielowi)
            patterns[j, best_var] = True

    return patterns
