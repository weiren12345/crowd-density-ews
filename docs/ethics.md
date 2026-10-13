# Ethics and data handling

Not legal advice, and not a substitute for your institution's process. Your
unit coordinator and your university's human research ethics committee decide
what applies. Ask early: approval timelines are measured in weeks, and this is
the item most likely to delay the project.

## Ask the coordinator first, before filming anything

Bring specifics, not a general question. Have ready:

- where you want to film, and whether it is public, semi-public (campus) or
  private property;
- whether faces will be identifiable at your camera distance and resolution;
- how long footage is retained, where it is stored, and who can access it;
- whether any data leaves the country or the institution (Colab uploads do);
- whether you intend to publish frames in the report or a presentation.

Ask explicitly whether your project needs full ethics review, qualifies for an
exemption, or is covered by an existing departmental approval. Get the answer
in writing and keep it with the project.

## The design already reduces the exposure

Points worth making in the application, because they are true of this system:

- **An elevated, downward camera sees the tops of heads.** At the angles this
  project needs, faces are mostly not visible. Say so, and show a sample frame.
- **The model is a detector, not a recogniser.** It outputs boxes. There is no
  identification, no re-identification across cameras, no biometric template,
  no attribute inference.
- **Track ids are per-clip and meaningless.** They exist to difference two
  positions 0.3 s apart and are discarded with the clip.
- **The outputs are aggregate.** Density, a velocity variance, a warning level.
  Nothing downstream of detection is about an individual.
- **Nothing needs to be retained.** The pipeline can run on a frame and keep
  only the CSV row. Say whether you are retaining footage and why.

## Practical measures

- **Minimise.** Record the shortest usable clips of the narrowest usable scene.
  Do not collect footage "in case it is useful".
- **Signage.** If you film in a space where people would not expect it, post a
  notice. Your institution may require specific wording.
- **Storage.** Keep raw footage out of the repository. `data/raw/` is
  gitignored for this reason. Keep it on an encrypted drive or an approved
  institutional store, not a personal cloud folder.
- **Sharing.** Do not upload identifiable footage to Colab or any third-party
  service. If you must train in the cloud, train on the public dataset
  (SCUT-HEAD) and evaluate locally on your own footage.
- **Publication.** Blur faces in any frame that goes in the report or slides,
  even if you judge them unidentifiable. Prefer the synthetic clip or an
  overlay-only figure for illustration.
- **Retention.** Decide a deletion date before you start, write it down, and
  honour it.
- **Withdrawal.** Have an answer ready for someone who asks you to stop
  filming or to delete footage containing them.

## Consider not filming people at all

Check whether you need your own footage to answer your research question. The
system can be demonstrated on:

- the synthetic clip, for the density / pressure / warning stack;
- SCUT-HEAD images, for detection;
- published crowd datasets whose licences permit research use.

Your own footage is most valuable for the domain-gap measurement in
`limitations.md` section 7. If that is the only thing it is for, you can scope
the filming much more tightly — a short clip of a moderately busy bottleneck,
not hours of recording — which makes the ethics application considerably
easier.

## A note on where this kind of system ends up

Crowd monitoring sits close to surveillance, and the distance between "density
in a region" and "tracking individuals" is a configuration change, not a
redesign. The project is honest if the writeup says so plainly: what the system
would need in order to become surveillance, and what in its design currently
prevents that. A paragraph on misuse potential is worth more than a paragraph
asserting good intentions.
