# Visual assets

`voice-evals-cover.png` is AI-generated artwork for the README. The waveform is an illustration, not a measured signal. The generated wordmark is a project identity concept.

The Python, Kaggle, PyPI, ElevenLabs and GitHub SVGs are sourced from [Simple Icons](https://simpleicons.org/) via `https://cdn.simpleicons.org/{slug}/8BE0B1`. They identify relevant technologies and do not imply endorsement. Brand marks remain subject to their owners' trademark guidelines; see [Simple Icons' disclaimer](https://github.com/simple-icons/simple-icons#disclaimer).

The generated English campaign images are included in [`social/`](social/): [GitHub social preview](social/github-social-preview.jpg), [LinkedIn launch image](social/linkedin-launch.png) and [DEV article cover](social/devto-cover.png). The social preview is ready to upload in repository settings; committing it does not change that setting. Publication drafts remain local under the ignored `out/launch-kit/` directory. Logo SVGs use a graphite background for readable contrast in GitHub's light and dark themes.

## Workflow diagram

[`evaluation-flow.svg`](evaluation-flow.svg) is rendered from [`evaluation-flow.mmd`](evaluation-flow.mmd). The vertical layout fits the README column, and embedding the SVG avoids GitHub's Mermaid viewer controls overlapping the diagram. Labels use SVG text instead of HTML for image compatibility. A white background preserves contrast in both GitHub themes.

Regenerate it from the repository root:

```bash
npx --yes --package @mermaid-js/mermaid-cli@12.0.0 mmdc -i docs/assets/evaluation-flow.mmd -o docs/assets/evaluation-flow.svg -b white --no-font-embed
```

## Video credits

`voice-evals-demo.mp4` is a 22-second, 1280 × 720, 30 fps H.264/AAC launch film authored with [brag](https://github.com/latent-spaces/brag) and [Hyperframes](https://github.com/heygen-com/hyperframes). The GIF is a silent 800-pixel-wide preview. The poster is the settled end card at 20 seconds, also baked into frame zero of the MP4.

The README embeds the looping GIF as its primary demo so it animates without a play click. This follows the image embed pattern used by [VHS](https://github.com/charmbracelet/vhs). GitHub's [animation accessibility preferences](https://docs.github.com/en/account-and-profile/how-tos/account-settings/managing-accessibility-settings) can disable automatic animation for an individual viewer.

An expandable section retains the GitHub attachment URL on its own paragraph for full-quality video with sound. That player requires a play click. Anonymous rendering and the resulting video download were verified against the local MP4 using SHA-256. Keep the attachment URL as a standalone paragraph; the repository MP4 and transcript provide alternatives.

- Music: [Happy Beats & Business Moves Vol. 12 by Sascha Ende](https://ende.app/en/song/12881-happy-), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: excerpted to 22 seconds, lowered in volume and faded at both ends. Source bundled by brag; [track and license details](https://ende.app/standard-license).
- Soft reveal sound: Kenney `interface/drop_001.ogg`, from [brag's bundled sound library](https://github.com/latent-spaces/brag/tree/main/skills/brag/assets/sfx). [Kenney interface sounds](https://kenney.nl/assets/interface-sounds), CC0.
- Display font: [Space Grotesk](https://github.com/floriankarsten/space-grotesk), SIL Open Font License 1.1.
- Artwork: built-in image generation. The waveform is illustrative.

Keep the music attribution with any public upload of the sound-enabled video. The soundtrack has its own license; the repository's Apache 2.0 software license does not replace third-party media licenses.
