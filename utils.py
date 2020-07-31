def params2name(params):
    results_name = ''
    for key in params.keys():
        results_name = results_name + key + '_' + str(params[key]) +'_'
    results_name = results_name[:-1]
    return results_name