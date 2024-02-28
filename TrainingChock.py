import log_tools
import emoji
import os

def unblock(args, optim_config, net_config, training_config):
    '''
        A function that force myself to read all the configs before I go afk
    '''
    log_tools.print_normal(args)
    log_tools.print_normal(optim_config)
    log_tools.print_normal(net_config)
    log_tools.print_normal(training_config)

    checker = input('DID YOU READ IT? [Y|N]')
    if checker == 'Y' or checker == 'y':
        log_tools.print_normal(emoji.emojize('We\'re ready to take off :rocket::rocket::rocket:, good luck', language='alias'))
    else:
        log_tools.print_warning(emoji.emojize('Houston, we have a problem :police_car_light::police_car_light::police_car_light:', language='alias'))
        exit()


def path_checker(path):
    '''
        Give a warning if the file already exists.
    '''
    if os.path.exists(path):
        log_tools.print_warning(f'Warning: {path} already exists!')
        checker = input('Are you sure you want to continue? [Y|N]')
        if checker == 'Y' or checker == 'y':
            log_tools.print_normal('Sure, as you wish')
        else:
            log_tools.print_warning('Oops')
            exit()
    else:
        log_tools.print_normal(f'OK: {path} is available.')
