classes = [
            'A1: drink water',
            'A2: eat meal',
            'A3: brush teeth',
            'A4: brush hair',
            'A5: drop',
            'A6: pick up',
            'A7: throw',
            'A8: sit down',
            'A9: stand up',
            'A10: clapping',
            'A11: reading',
            'A12: writing',
            'A13: tear up paper',
            'A14: put on jacket',
            'A15: take off jacket',
            'A16: put on a shoe',
            'A17: take off a shoe',
            'A18: put on glasses',
            'A19: take off glasses',
            'A20: put on a hat or cap',
            'A21: take off a hat or cap',
            'A22: cheer up',
            'A23: hand waving',
            'A24: kicking something',
            'A25: reach into pocket',
            'A26: hopping',
            'A27: jump up',
            'A28: phone call',
            'A29: play with phone or tablet',
            'A30: type on a keyboard',
            'A31: point to something',
            'A32: taking a selfie',
            'A33: check time (from watch)',
            'A34: rub two hands',
            'A35: nod head/bow',
            'A36: shake head',
            'A37: wipe face',
            'A38: salute',
            'A39: put palms together',
            'A40: cross hands in front',
            'A41: sneeze or cough',
            'A42: staggering',
            'A43: falling down',
            'A44: headache',
            'A45: chest pain',
            'A46: back pain',
            'A47: neck pain',
            'A48: nausea/vomiting',
            'A49: fan self',
            'A50: punch/slap',
            'A51: kicking',
            'A52: pushing',
            'A53: pat on back',
            'A54: point finger',
            'A55: hugging',
            'A56: giving object',
            'A57: touch pocket',
            'A58: shaking hands',
            'A59: walking towards',
            'A60: walking apart']
        

train_x = [
            'put on glasses',
            'clapping',
            'check time (from watch)',
            'chest pain',
            'giving object',
            'put on glasses',
            'staggering',
            'sit down',
            'neck pain',
            'kicking',
            'chest pain',
            'wipe face',
            'nausea/vomiting',
            'pushing',
            'take off a hat/cap',
            'hugging',
            'sit down',
            'hand waving',
            'put on a hat/cap',
            'pick up',
            'giving object',
            'pick up',
            'tear up paper',
            'put on jacket',
            'eat meal',
            'wipe face',
            'touch pocket',
            'kicking something',
            'punch/slap',
            'pushing',
            'sit down',
            'put on a hat/cap',
            'sit down',
            'take off jacket',
            'nausea/vomiting',
            'hand waving',
            'point to something',
            'nausea/vomiting',
            'reach into pocket',
            'point to something',
            'put on a shoe',
            'put on a shoe',
            'nod head/bow',
            'shake head',
            'put on glasses',
            'writing',
            'wipe face',
            'cheer up']

ntu_joints = [
                'pelvis',
                'torso',
                'neck',
                'head',
                'left shoulder',
                'left elbow',
                'left arm',
                'left wrist',
                'right shoulder',
                'right elbow',
                'right arm',
                'right wrist',
                'left hip',
                'left knee',
                'left ankle',
                'left foot',
                'right hip',
                'right knee',
                'right ankle',
                'right foot',
                'chest',
                'left hand',
                'left thumb',
                'right hand',
                'right thumb'
            ]

joint_color = {
                'pelvis'        : '#000000',
                'torso'         : '#000000',
                'neck'          : '#000000',
                'head'          : '#000000',
                'left shoulder' : '#4d0000',
                'left elbow'    : '#7a0000',
                'left arm'      : '#ab0000',
                'left wrist'    : '#ff0000',
                'right shoulder': '#613500',
                'right elbow'   : '#9c5702',
                'right arm'     : '#c96f00',
                'right wrist'   : '#ff8c00',
                'left hip'      : '#3d4000',
                'left knee'     : '#747a00',
                'left ankle'    : '#b3bd00',
                'left foot'     : '#f2ff00',
                'right hip'     : '#003d0b',
                'right knee'    : '#006613',
                'right ankle'   : '#02ab21',
                'right foot'    : '#00ff2f',
                'chest'         : '#000000',
                'left hand'     : '#ff0000',
                'left thumb'    : '#ff0000',
                'right hand'    : '#ff8c00',
                'right thumb'   : '#ff8c00'
            }
'''
 0: pelvis
 1: torso
 2: neck
 3: head
 4: left shoulder
 5: left elbow
 6: left arm
 7: left wrist
 8: right shoulder
 9: right elbow
10: right arm
11: right wrist
12: left hip
13: left knee
14: left ankle
15: left foot
16: right hip
17: right knee
18: right ankle
19: right foot
20: chest
21: left hand
22: left thumb
23: right hand
24: right thumb

 A1: drink water
 A2: eat meal
 A3: brush teeth
 A4: brush hair
 A5: drop
 A6: pick up
 A7: throw
 A8: sit down
 A9: stand up
A10: clapping
A11: reading
A12: writing
A13: tear up paper
A14: put on jacket
A15: take off jacket
A16: put on a shoe
A17: take off a shoe
A18: put on glasses
A19: take off glasses
A20: put on a hat/cap
A21: take off a hat/cap
A22: cheer up
A23: hand waving
A24: kicking something
A25: reach into pocket
A26: hopping
A27: jump up
A28: phone call
A29: play with phone/tablet
A30: type on a keyboard
A31: point to something
A32: taking a selfie
A33: check time (from watch)
A34: rub two hands
A35: nod head/bow
A36: shake head
A37: wipe face
A38: salute
A39: put palms together
A40: cross hands in front
A41: sneeze/cough
A42: staggering
A43: falling down
A44: headache
A45: chest pain
A46: back pain
A47: neck pain
A48: nausea/vomiting
A49: fan self
A50: punch/slap
A51: kicking
A52: pushing
A53: pat on back
A54: point finger
A55: hugging
A56: giving object
A57: touch pocket
A58: shaking hands
A59: walking towards
A60: walking apart
'''
