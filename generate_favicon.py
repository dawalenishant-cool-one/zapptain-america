import numpy as np
import matplotlib.pyplot as plt

def main():
    t = np.linspace(0, 1, 300)
    
    # Envelope to taper oscillations to 0 at the endpoints
    env = np.sin(np.pi * t)
    
    # Segment 1: top bar (left-to-right, y around 1)
    x1 = t
    y1 = 1.0 + 0.12 * np.sin(35 * t) * env

    # Segment 2: diagonal (top-right to bottom-left, y from 1 to 0, x from 1 to 0)
    x2 = 1.0 - t
    y2 = 1.0 - t
    normal_x = 1.0 / np.sqrt(2)
    normal_y = -1.0 / np.sqrt(2)
    osc = 0.12 * np.sin(35 * t) * env
    x2_osc = x2 + osc * normal_x
    y2_osc = y2 + osc * normal_y

    # Segment 3: bottom bar (left-to-right, y around 0)
    x3 = t
    y3 = 0.0 + 0.12 * np.sin(35 * t) * env

    # Concatenate them
    x = np.concatenate([x1, x2_osc, x3])
    y = np.concatenate([y1, y2_osc, y3])

    fig, ax = plt.subplots(figsize=(2, 2), dpi=128)
    
    # Glow effect by layering plots
    ax.plot(x, y, color='#c084fc', linewidth=7, alpha=0.25, solid_capstyle='round')
    ax.plot(x, y, color='#818cf8', linewidth=4, alpha=0.6, solid_capstyle='round')
    ax.plot(x, y, color='#ffffff', linewidth=1.5, alpha=1.0, solid_capstyle='round')

    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Tight limits to center the icon
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)

    plt.savefig('favicon.png', bbox_inches='tight', pad_inches=0, transparent=True, dpi=128)
    print("Favicon generated successfully.")

if __name__ == '__main__':
    main()
