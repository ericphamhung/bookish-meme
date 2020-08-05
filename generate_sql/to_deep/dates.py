from dateutil.parser import parse


def get_all_dates(string):
    try:
        tstring = parse(string, fuzzy_with_tokens = True)
    except:
        return None
    if len(tstring) == 2:
        dates = [tstring[0]]
        try:
            ss = string.replace(tstring[1][0], '')
            td = get_all_dates(ss.split(tstring[1][-1])[1])
            if td is not None:
                dates.extend(td)
        except:
            pass
        
    return dates


if __name__ == '__main__':
    date_string = 'It was between 2010-01-10 and 2011-02-04'
    print(get_all_dates(date_string))
