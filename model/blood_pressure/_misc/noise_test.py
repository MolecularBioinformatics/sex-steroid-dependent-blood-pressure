import noise
import numpy as np
import matplotlib.pyplot as plt
import colorednoise as cn

exponent = 2
samples = 156000

### pink noise generation
pink_noise = cn.powerlaw_psd_gaussian(exponent, samples)  # Pink noise
### perlin noise generation
perlin_noise = [noise.pnoise1(x) for x in np.linspace(0, 1000, samples)]
perlin_noise2 = [noise.pnoise1(x) for x in np.linspace(0, 1440*180, 6*1440*180)]

### fft noise generation
smoothness = 0.05  # lower = smoother (attenuates high freq)
np.random.seed(0)

# Random phases and amplitudes in frequency domain
amplitudes = np.random.randn(samples//2 + 1) * np.exp(-smoothness * np.arange(samples//2 + 1))
phases = np.exp(1j * 2 * np.pi * np.random.rand(samples//2 + 1))
spectrum = amplitudes * phases

# Make it Hermitian symmetric → real signal after IFFT
spectrum_full = np.concatenate([spectrum, np.conj(spectrum[-2:0:-1])])

# Inverse FFT to get time-domain signal
fft_noise = np.fft.ifft(spectrum_full).real

# Normalize to [-1, 1]
fft_noise /= np.max(np.abs(fft_noise))

###
fig, axes = plt.subplots(3, 1, figsize=(6, 8))

axes[0].plot(perlin_noise, label='Perlin Noise')
axes[0].set_title('Perlin Noise')

axes[0].plot(perlin_noise2, label='Perlin Noise 2')
axes[0].set_title('Perlin Noise 2')

axes[1].plot(pink_noise)
axes[1].set_title('Pink Noise')

axes[2].plot(fft_noise)
axes[2].set_title('FFT-based Noise')

plt.tight_layout()
plt.legend(loc='upper right')
plt.show()

