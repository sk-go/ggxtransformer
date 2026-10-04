#!/usr/bin/env python3
"""
valenz_korpus.py
Erzeugt und validiert 100 Minimalpaare für das Valenz-Inkongruenz-Experiment (PRAEREG_5_valenz.md).

Bedingungen:
  K1: Konsistente Handlung (Art ✓, Folge ✓)
  K2: Paradoxe Handlung (Art ✗ [Valenzbruch: Fürsorge/Schutz/Freundlichkeit -> Gewalt/Zerstörung], Folge ✓)
  K3: Neutrale semantische Inkongruenz (Art ~ [Kategoriefehler], Folge ✓)
"""

import json
from pathlib import Path

TEMPLATES = [
    # 1-10: Fürsorge / Familie / Kinder
    ("The devoted mother gently held her sick child and", "comforted", "strangled", "measured", "its fragile body."),
    ("The loving father watched his daughter take her first steps and", "cheered", "tripped", "compiled", "her enthusiastic progress."),
    ("The kind babysitter read a bedtime story to the infant and", "soothed", "smothered", "indexed", "its quiet tears."),
    ("The caring nurse sat beside the dying elder and", "relieved", "prolonged", "calculated", "his agonizing pain."),
    ("The gentle sister leaned down to the frightened boy and", "embraced", "struck", "inspected", "his trembling shoulders."),
    ("The affectionate mother tucked her shivering son into bed and", "kissed", "bitten", "inventoried", "his pale forehead."),
    ("The proud grandfather picked up his laughing granddaughter and", "hugged", "dropped", "calibrated", "her tiny waist."),
    ("The protective mother bear guided her playful cubs to the river and", "shielded", "drowned", "cataloged", "their clumsy movement."),
    ("The caring nanny prepared a warm bath for the toddler and", "washed", "scalded", "tabulated", "his delicate skin."),
    ("The doting uncle bought a wooden toy for the smiling orphan and", "delighted", "choked", "alphabetized", "his eager face."),

    # 11-20: Medizin / Heilung / Therapie
    ("The dedicated surgeon approached the unconscious patient and", "repaired", "mutilated", "cataloged", "the damaged organ."),
    ("The compassionate doctor cleaned the infected wound and", "bandaged", "burned", "tabulated", "the swollen tissue."),
    ("The gentle therapist listened to the trauma victim and", "healed", "mocked", "formatted", "her broken spirit."),
    ("The hospice worker offered warm soup to the starved man and", "nourished", "poisoned", "audited", "his failing strength."),
    ("The paramedic rushed to the bleeding crash survivor and", "stabilized", "crushed", "summarized", "his broken limbs."),
    ("The kind dentist noticed the terrified boy shaking and", "calmed", "pierced", "counted", "his nervous breath."),
    ("The caring ophthalmologist inspected the injured eye and", "cured", "blinded", "audited", "the damaged cornea."),
    ("The pediatric nurse carefully administered the life-saving vaccine and", "protected", "infected", "balanced", "the crying baby."),
    ("The dedicated psychiatrist spoke softly to the suicidal youth and", "restored", "destroyed", "itemized", "his fading hope."),
    ("The physical therapist supported the paralyzed veteran and", "guided", "dropped", "quantified", "his trembling legs."),

    # 21-30: Schutz / Treue / Freundschaft
    ("The loyal bodyguard stood before the threatened diplomat and", "shielded", "shot", "balanced", "his vulnerable back."),
    ("The faithful friend listened to the secret confession and", "forgave", "betrayed", "counted", "the painful truth."),
    ("The honest partner received the shared company profits and", "protected", "embezzled", "converted", "the collective wealth."),
    ("The sworn guardian guarded the sleeping village gates and", "defended", "torched", "classified", "the peaceful homes."),
    ("The devoted companion helped his blind master across the street and", "guided", "shoved", "sketched", "his cautious steps."),
    ("The loyal soldier carried his wounded comrade through gunfire and", "saved", "stabbed", "computed", "his fading pulse."),
    ("The trusted confidant held the diary containing deep secrets and", "honored", "exposed", "cataloged", "every private confession."),
    ("The faithful sheepdog watched over the flock at dusk and", "protected", "slaughtered", "numbered", "the vulnerable lambs."),
    ("The childhood companion stood beside his crying friend and", "comforted", "ridiculed", "graphed", "his bitter sorrow."),
    ("The dedicated watchman patrolled the quiet museum corridors and", "secured", "vandalized", "itemized", "the precious artifacts."),

    # 31-40: Rettung / Notfall / Katastrophe
    ("The brave firefighter entered the blazing nursery and", "rescued", "trapped", "calibrated", "the suffocating baby."),
    ("The noble knight pulled the drowning peasant from the river and", "revived", "drowned", "factored", "his freezing lungs."),
    ("The mountain guide reached the stranded climber on the cliff and", "secured", "pushed", "itemized", "his slipping harness."),
    ("The rescue diver found the lost boy in the submerged cave and", "freed", "caged", "sorted", "his tangled oxygen lines."),
    ("The relief worker handed fresh bottled water to the thirsty refugee and", "quenched", "scalded", "logged", "his burning throat."),
    ("The coast guard captain spotted the sinking fishing vessel and", "towed", "rammed", "tabulated", "the leaking hull."),
    ("The avalanche rescue team dug through the freezing snowpack and", "warmed", "buried", "audited", "the trapped survivor."),
    ("The brave civilian grabbed the woman falling from the bridge and", "hoisted", "hurled", "compiled", "her hanging body."),
    ("The emergency pilot steered the burning aircraft toward the runway and", "landed", "crashed", "calculated", "the crowded vessel."),
    ("The bomb technician gently defused the ticking explosive device and", "neutralized", "detonated", "indexed", "the deadly mechanism."),

    # 41-50: Lehre / Weisheit / Mentorschaft
    ("The patient teacher noticed the struggling student and", "encouraged", "humiliated", "multiplied", "his nervous attempt."),
    ("The wise mentor observed the young apprentice's mistake and", "corrected", "punished", "translated", "his awkward gesture."),
    ("The gracious host welcomed the exhausted travelers inside and", "warmed", "starved", "plotted", "their weary limbs."),
    ("The generous benefactor heard about the poor orphan's dream and", "funded", "destroyed", "charted", "his future schooling."),
    ("The caring grandmother knitted a warm wool scarf for her grandson and", "wrapped", "choked", "measured", "his shivering neck."),
    ("The master artisan guided the novice's unsteady chisel and", "shaped", "ruined", "tallied", "the marble sculpture."),
    ("The school counselor listened to the bullied teenager and", "supported", "insulted", "cataloged", "her lonely struggle."),
    ("The elder scholar shared ancient manuscripts with the student and", "enlightened", "misled", "indexed", "his searching intellect."),
    ("The music teacher tuned the student's damaged violin and", "harmonized", "snapped", "digitized", "its delicate strings."),
    ("The sports coach hugged the defeated athlete after the match and", "praised", "scorned", "ranked", "his brave performance."),

    # 51-60: Gerechtigkeit / Herrschaft / Versöhnung
    ("The peacemaker stepped between the hostile duelists and", "reconciled", "slaughtered", "quantified", "their bitter hatred."),
    ("The benevolent king listened to the kneeling rebel's plea and", "pardoned", "beheaded", "indexed", "his desperate crime."),
    ("The forgiving victim looked into the eyes of the repentant thief and", "blessed", "stabbed", "averaged", "his weeping face."),
    ("The righteous judge weighed the evidence for the innocent man and", "acquitted", "condemned", "categorized", "his tarnished reputation."),
    ("The honest officer found the lost wallet with thousand dollars and", "returned", "stole", "processed", "the forgotten fortune."),
    ("The compassionate governor visited the impoverished flood zone and", "rebuilt", "looted", "audited", "the destroyed towns."),
    ("The noble ambassador presented the mutual peace treaty and", "signed", "burned", "transcribed", "the historic accord."),
    ("The humble monk received the wounded enemy soldier inside and", "sheltered", "poisoned", "cataloged", "his bleeding body."),
    ("The fair arbitrator examined both sides of the bitter dispute and", "resolved", "inflamed", "tallied", "their conflicting demands."),
    ("The selfless monarch opened the royal granaries during famine and", "fed", "starved", "budgeted", "the desperate citizens."),

    # 61-70: Mensch & Tier / Natur
    ("The compassionate monk offered a bowl of rice to the starving stray dog and", "fed", "kicked", "audited", "its hungry mouth."),
    ("The dedicated wildlife vet pulled the metal snare from the trapped deer and", "healed", "strangled", "tallied", "its injured leg."),
    ("The loving owner gently stroked the trembling elderly cat and", "comforted", "suffocated", "measured", "its aching joints."),
    ("The park ranger discovered the abandoned eagle chick in the woods and", "nourished", "crushed", "weighed", "its fragile wings."),
    ("The gentle farmer released the caught bird from the chicken coop and", "freed", "decapitated", "counted", "its fluttery feathers."),
    ("The dolphin rescuer held the beached whale in shallow water and", "hydrated", "cut", "cataloged", "its dry skin."),
    ("The caring gardener pruned the sick fruit tree in early spring and", "revitalized", "hacked", "mapped", "its tender branches."),
    ("The horse trainer patiently trained the abused young stallion and", "tamed", "whipped", "calibrated", "its fearful temperament."),
    ("The marine biologist cleaned the toxic oil from the seal pup and", "purified", "scraped", "tabulated", "its soaked fur."),
    ("The boy found a injured turtle on the busy highway and", "carried", "smashed", "classified", "its painted shell."),

    # 71-80: Bündnis / Kollegialität / Kooperation
    ("The research scientist discovered the cure for the pandemic and", "shared", "suppressed", "converted", "the medical breakthrough."),
    ("The cooperative coworker stayed late to help his sick colleague and", "finished", "erased", "logged", "the crucial presentation."),
    ("The mountaineering partner held the safety rope firmly and", "anchored", "cut", "balanced", "his climbing companion."),
    ("The reliable architect inspected the structural load bearing pillar and", "reinforced", "demolished", "computed", "the weak foundation."),
    ("The skilled mechanic repaired the family's broken brake system and", "secured", "tampered", "charted", "their road safety."),
    ("The honest inspector checked the airplane passenger door and", "sealed", "unlocked", "documented", "the pressurized cabin."),
    ("The fellow explorer found his lost partner in the arctic storm and", "warmed", "abandoned", "numbered", "his frozen limbs."),
    ("The shipmate pulled his exhausted friend from the icy surf and", "revived", "strangled", "estimated", "his shivering torso."),
    ("The sympathetic employer noticed his employee's personal tragedy and", "supported", "fired", "calculated", "his grieving household."),
    ("The generous team captain passed the winning ball to the rookie and", "celebrated", "mocked", "timed", "his victorious moment."),

    # 81-90: Gastfreundschaft / Zuflucht / Wohlwollen
    ("The kind innkeeper opened his warm tavern door to the frozen beggar and", "welcomed", "beat", "itemized", "his shivering presence."),
    ("The village elder invited the persecuted strangers into the circle and", "protected", "betrayed", "surveyed", "their sacred rights."),
    ("The monastery kitchen distributed fresh warm bread to the homeless and", "satisfied", "poisoned", "logged", "their desperate hunger."),
    ("The compassionate landlord saw the destitute family with sick infant and", "housed", "evicted", "averaged", "their fragile belongings."),
    ("The gracious neighbor brought hot homemade soup to the grieving widow and", "comforted", "berated", "calibrated", "her solitary heart."),
    ("The humanitarian worker opened the refugee camp gates at midnight and", "embraced", "expelled", "registered", "the fleeing crowd."),
    ("The hospitable farmer let the weary wanderer sleep in the dry barn and", "sheltered", "attacked", "inventoried", "his tired head."),
    ("The church community collected clothes for the disaster victims and", "donated", "shredded", "sorted", "the collected garments."),
    ("The friendly villager pulled the stranger's cart from the muddy ditch and", "rescued", "overturned", "weighed", "his heavy baggage."),
    ("The generous merchant forgave the honest carpenter's unpaid debt and", "relieved", "bankrupted", "audited", "his modest business."),

    # 91-100: Trost / Seelsorge / Weihe
    ("The solemn chaplain knelt beside the weeping soldier on the battlefield and", "prayed", "cursed", "tallied", "his dying spirit."),
    ("The empathetic rabbi listened to the grieving widower's sorrow and", "comforted", "scorned", "transcribed", "his deep despair."),
    ("The peaceful priest offered communion to the trembling prisoner and", "blessed", "struck", "indexed", "his weary soul."),
    ("The gentle counselor dried the tears of the grieving mother and", "embraced", "ridiculed", "evaluated", "her shattered life."),
    ("The loving poet wrote an elegy for his departed beloved and", "immortalized", "defamed", "analyzed", "her sacred memory."),
    ("The devoted monk lit a candle for the departed soul and", "honored", "desecrated", "measured", "the holy shrine."),
    ("The choir director guided the children's voices in harmony and", "uplifted", "silenced", "digitized", "their joyful song."),
    ("The wise elder placed a garland of flowers upon the hero's tomb and", "revered", "trampled", "tallied", "his heroic sacrifice."),
    ("The caring volunteer read poetry aloud to the blind orphan and", "brightened", "tormented", "indexed", "his quiet isolation."),
    ("The holy saint touched the suffering leper with bare hands and", "healed", "infected", "formatted", "his decaying flesh.")
]

def generate_stimuli():
    items = []
    for idx, (prefix, k1, k2, k3, suffix) in enumerate(TEMPLATES):
        items.append({
            "id": idx + 1,
            "prefix": prefix,
            "suffix": suffix,
            "targets": {
                "K1_konsistent": k1,
                "K2_paradox": k2,
                "K3_neutral": k3
            },
            "sentences": {
                "K1": f"{prefix} {k1} {suffix}",
                "K2": f"{prefix} {k2} {suffix}",
                "K3": f"{prefix} {k3} {suffix}"
            }
        })
    return items

def main():
    out_dir = Path("korpus_valenz")
    out_dir.mkdir(parents=True, exist_ok=True)
    stimuli = generate_stimuli()
    
    out_path = out_dir / "stimuli.json"
    out_path.write_text(json.dumps(stimuli, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(stimuli)} Minimalpaar-Templates generiert -> {out_path}")
    
    for s in stimuli[:3]:
        print(f"\n[ID {s['id']}]")
        print(f"  K1 (Konsistent): {s['sentences']['K1']}")
        print(f"  K2 (Paradox):    {s['sentences']['K2']}")
        print(f"  K3 (Neutral):    {s['sentences']['K3']}")

if __name__ == "__main__":
    main()
