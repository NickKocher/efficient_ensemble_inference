
IDLE_POWER_BOOST=0.09374511555335349
IDLE_POWER_NO_BOOST=0.06340445107669676


def get_idle_power(boost=True):
    if boost:
        return IDLE_POWER_BOOST
    else:
        return IDLE_POWER_NO_BOOST