"""
@since 20170517
https://en.wikipedia.org/wiki/Homomorphic_filtering
based on this excellent answer: http://stackoverflow.com/a/24732434/2692914
"""
import numpy as np
import scipy.fftpack


def hfilter(gray, M, N, sigma, low_gamma=0.3, high_gamma=1.5):
    rows, cols = gray.shape

    imgLog = np.log1p(np.array(gray, dtype="float") / 255)

    # M = 2 * rows + 1
    # N = 2 * cols + 1
    # sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # divides la imagen en alta frecuencia y baja frecuencia
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # las unes con diferentes coeficientes
    gamma1 = low_gamma
    if low_gamma != 1.0:
        Ioutlow_gamma = gamma1 * Ioutlow[0:rows, 0:cols]
    else:
        Ioutlow_gamma = Ioutlow[0:rows, 0:cols]

    gamma2 = high_gamma
    if high_gamma != 1.0:
        Iouthigh_gamma = gamma2 * Iouthigh[0:rows, 0:cols]
    else:
        Iouthigh_gamma = Iouthigh[0:rows, 0:cols]

    Iout = Ioutlow_gamma + Iouthigh_gamma

    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))

    Iolg = np.expm1(Ioutlow_gamma)
    Iolg = (Iolg - np.min(Iolg)) / (np.max(Iolg) - np.min(Iolg))

    Iohg = np.expm1(Iouthigh_gamma)
    Iohg = (Iohg - np.min(Iohg)) / (np.max(Iohg) - np.min(Iohg))

    return np.array(255 * Ihmf, dtype="uint8"), np.array(255 * Iolg, dtype="uint8"), np.array(255 * Iohg, dtype="uint8")


def hfilter2(gray, M, N, sigma, low_gamma=0.3, high_gamma=1.5):
    rows, cols = gray.shape

    imgLog = np.log1p(np.array(gray, dtype="float") / 255)

    # M = 2 * rows + 1
    # N = 2 * cols + 1
    # sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # divides la imagen en alta frecuencia y baja frecuencia
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # las unes con diferentes coeficientes
    gamma1 = low_gamma
    if low_gamma != 1.0:
        Ioutlow_gamma = gamma1 * Ioutlow[0:rows, 0:cols]
    else:
        Ioutlow_gamma = Ioutlow[0:rows, 0:cols]

    gamma2 = high_gamma
    if high_gamma != 1.0:
        Iouthigh_gamma = gamma2 * Iouthigh[0:rows, 0:cols]
    else:
        Iouthigh_gamma = Iouthigh[0:rows, 0:cols]

    Iout = Iouthigh_gamma - Ioutlow_gamma

    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))

    Iolg = np.expm1(Ioutlow_gamma)
    Iolg = (Iolg - np.min(Iolg)) / (np.max(Iolg) - np.min(Iolg))

    Iohg = np.expm1(Iouthigh_gamma)
    Iohg = (Iohg - np.min(Iohg)) / (np.max(Iohg) - np.min(Iohg))

    return np.array(255 * Ihmf, dtype="uint8"), np.array(255 * Iolg, dtype="uint8"), np.array(255 * Iohg, dtype="uint8")
