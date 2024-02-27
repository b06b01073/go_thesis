import log_tools
import emoji

def unblock(args, optim_config, net_config, training_config):
    '''
        A function that force myself to read all the configs before I go afk
    '''
    log_tools.print_normal(args)
    log_tools.print_normal(optim_config)
    log_tools.print_normal(net_config)
    log_tools.print_normal(training_config)

    checker = input('DID YOU READ IT? [Y|N]')
    if not checker == 'Y' and not checker == 'y':
        print(emoji.emojize('Houston, we\'ve got a problem :police_car_light::police_car_light::police_car_light:', language='alias'))
        exit()
    else:
        print(emoji.emojize('We\'re ready to take off :rocket::rocket::rocket:, good luck', language='alias'))
