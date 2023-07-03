import numpy as np

__all__ = ['extractObservation']


powsof2 = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
)


def decode(estate):
    """
    decodes the encoded state estate, which is a string of 61 chars
    """
    #    powsof2 = (1, 2, 4, 8, 16, 32, 64, 128)
    dstate = np.empty(shape=(22, 22), dtype=np.int)
    for i in range(22):
        for j in range(22):
            dstate[i, j] = 2
    row = 0
    col = 0
    totalBitsDecoded = 0
    reqSize = 31
    assert (
        len(estate) == reqSize
    ), f'Error in data size given {len(estate)}! Required: {reqSize} \n data: {estate}'
    check_sum = 0
    for i in range(len(estate)):
        cur_char = estate[i]
        if ord(cur_char) != 0:
            check_sum += ord(cur_char)
        for j in range(16):
            totalBitsDecoded += 1
            if col > 21:
                row += 1
                col = 0
            if (int(powsof2[j]) & int(ord(cur_char))) != 0:
                dstate[row, col] = 1
            else:
                dstate[row, col] = 0
            col += 1
            if totalBitsDecoded == 484:
                break
    print(f'totalBitsDecoded: {totalBitsDecoded}')
    return dstate, check_sum


def extractObservation(data: bytes):
    """
    parse the array of strings and return array 22 by 22 of doubles
    """
    data = data.decode()
    # observation lenght: 487
    levelScene = np.empty(shape=(22, 22), dtype=np.int)
    enemiesFloats = []
    dummy = 0
    if data[0] == 'E':
        mayMarioJump = data[1] == '1'
        isMarioOnGround = data[2] == '1'
        levelScene, check_sum_got = decode(data[3:34])
        check_sum_recv = int(data[34:])
        if check_sum_got != check_sum_recv:
            print(f'Error check_sum! got {check_sum_got} != recv {check_sum_recv}')

        return (mayMarioJump, isMarioOnGround, levelScene)
    data = data.split(' ')
    if data[0] == 'FIT':
        status = int(data[1])
        distance = float(data[2])
        timeLeft = int(data[3])
        marioMode = int(data[4])
        coins = int(data[5])

        return status, distance, timeLeft, marioMode, coins

    elif data[0] == 'O':
        mayMarioJump = data[1] == 'true'
        isMarioOnGround = data[2] == 'true'
        k = 0
        for i in range(22):
            for j in range(22):
                levelScene[i, j] = int(data[k + 3])
                k += 1
        k += 3
        try:
            float_x = float(data[k])
        except ValueError:
            float_x = float(data[k][:-2])

        try:
            float_y = float(data[k + 1])
        except ValueError:
            float_y = float(data[k + 1][:-2])
        marioFloats = (float_x, float_y)
        k += 2
        while k < len(data):
            enemiesFloats.append(float(data[k]))
            k += 1

        return (
            mayMarioJump,
            isMarioOnGround,
            marioFloats,
            enemiesFloats,
            levelScene,
            dummy,
        )

    else:
        raise ValueError('Wrong format or corrupted observation...')
