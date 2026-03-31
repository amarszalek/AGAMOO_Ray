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

    for i in range(nvars):
        var_data = front[:, i]

        # Zabezpieczenie przed brakiem wariancji
        if np.std(var_data) == 0:
            probs = np.ones(nobjs) / nobjs  # Równe prawdopodobieństwo dla każdego gracza
        else:
            corrs = np.zeros(nobjs)
            for j in range(nobjs):
                obj_data = front_eval[:, j]
                if np.std(obj_data) > 0:
                    # Liczymy bezwzględną korelację (nie interesuje nas znak, tylko siła)
                    corrs[j] = np.abs(np.corrcoef(var_data, obj_data)[0, 1])

            # Normalizacja korelacji do wektora prawdopodobieństw [0, 1]
            sum_corrs = np.sum(corrs)
            if sum_corrs > 0:
                probs = corrs / sum_corrs
            else:
                # Jeśli z jakiegoś powodu wszystkie korelacje wynoszą 0
                probs = np.ones(nobjs) / nobjs

        # KLUCZOWA ZMIANA: Losowanie gracza (kryterium) z użyciem wyliczonych prawdopodobieństw
        chosen_obj = np.random.choice(nobjs, p=probs)
        patterns[chosen_obj, i] = True

    # Opcjonalne zabezpieczenie: Upewnijmy się, że żaden gracz nie został z pustą ręką
    empty_players = np.where(~np.any(patterns, axis=1))[0]

    for j in empty_players:
        # Szukamy "dawców" - graczy, którzy mają więcej niż 1 gen
        donor_candidates = np.where(np.sum(patterns, axis=1) > 1)[0]

        if len(donor_candidates) == 0:
            # Sytuacja ekstremalna (np. nvars < nobjs).
            break

        # Wybieramy losowego dawcę z puli bogatych
        donor = np.random.choice(donor_candidates)

        # Pobieramy listę genów należących do tego dawcy i losujemy jeden z nich
        donor_genes = np.where(patterns[donor])[0]
        stolen_gene = np.random.choice(donor_genes)

        # Bezpieczny transfer genu
        patterns[donor, stolen_gene] = False
        patterns[j, stolen_gene] = True

    return patterns
