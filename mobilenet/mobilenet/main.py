from mobilenet import MobileNetMod

classMap = {0: 'hand', 1: 'ok', 2: 'paper', 3: 'rock',
            4: 'scissors', 5: 'the-finger', 6: 'thumbdown', 7: 'thumbup'}
train_path = '../data/train'
test_path = '../data/test'
net = MobileNetMod(classMap)

for i in range(1):
    # net.train(train_path, test_path)
    net.test(test_path)
