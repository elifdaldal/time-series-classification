#Dataset Loader
c_activity_category = 'state' # 'lhb' , 'bhb'
c_activity_count = 6 #61
c_sensors = ['Accelerometer','Gyroscope','LinearAcceleration','Gravity','MSAccelerometer','MSGyroscope']

#Codebook Approach
c_window_length = 20
c_overlap_length = c_window_length // 2   
c_cluster_number = 6

c_labels = {0:'Bending',1:'Lying',2:'Sitting',3:'Squatting',4:'Standing',5:'Walking',6:'Bring',7:'CleanFloor',8:'CleanSurface',9:'CloseBigBox',10:'CloseDoor',11:'CloseDrawer',12:'CloseLidByRotate',13:'CloseOtherLid',14:'CloseSmallBox',16:'Drink',17:'DryOffHand',18:'DryOffByHandShake',19:'EatSmall',20:'Gargle',21:'GettingUp',22:'Hang',23:'LyingDown',24:'OpenBag',25:'OpenBigBox',26:'OpenDoor',27:'OpenDrawer',28:'OpenLidByRotate',29:'OpenOtherLid',30:'OpenSmallBox',32:'PlugIn',33:'PressByGrasp',34:'PressFromTop',35:'PressSwitch',36:'PutFromBottle',37:'PutFromTapWater',38:'PutHighPosition',39:'PutOnFloor',40:'Read',41:'Rotate',42:'RubHands',43:'ScoopPut',44:'SittingDown',45:'SquattingDown',46:'StandingUp',47:'StandUpFromSquatting',48:'TakeFromFloor',49:'TakeFromHighPosition',50:'TakeOffJacket',51:'TakeOut',52:'TalkByTelephone',53:'ThrowOut',54:'ThrowOutWater',55:'TouchSmartPhoneScreen',56:'Type',57:'Unhang',58:'Unplug',59:'WearJacket',60:'Write'}

#svc
c_cv_svc = 10
c_tuned_parameters_svc = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1], 'C': [0.001, 0.01, 0.1, 1, 10]}]

#rfc
c_cv_rfc = 10
c_tuned_parameters_rfc = {"max_depth": range(2,15), "n_estimators" : range(40,500,20), "criterion": ["gini", "entropy"]}
