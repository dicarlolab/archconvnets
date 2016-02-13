import numpy as np

syn_cats = np.zeros(91,dtype='int')

N_ANIMALS = 17
N_PLANES = 8
N_FACES = 10
N_CHAIRS = 9
N_TABLES = 15
N_PLANTS = 16
N_BODIES = 16

syn_cats[N_ANIMALS:N_ANIMALS+N_PLANES] = 1 # planes
syn_cats[N_ANIMALS+N_PLANES:N_ANIMALS+N_PLANES+N_FACES] = 2 # faces
syn_cats[N_ANIMALS+N_PLANES+N_FACES:N_ANIMALS+N_PLANES+N_FACES+N_CHAIRS] = 3 # chairs
syn_cats[N_ANIMALS+N_PLANES+N_FACES+N_CHAIRS:N_ANIMALS+N_PLANES+N_FACES+N_CHAIRS+N_TABLES] = 4 # tables
syn_cats[N_ANIMALS+N_PLANES+N_FACES+N_CHAIRS+N_TABLES:N_ANIMALS+N_PLANES+N_FACES+N_CHAIRS+N_TABLES+N_PLANTS] = 5 # plants
syn_cats[N_ANIMALS+N_PLANES+N_FACES+N_CHAIRS+N_TABLES+N_PLANTS:N_ANIMALS+N_PLANES+N_FACES+N_CHAIRS+N_TABLES+N_PLANTS+N_BODIES] = 6 # bodies

objs = ['shorthair_cat', 'lynx', 'leopard', 'doberman', 'weimaraner',
               'hedgehog', 'hare', 'fieldmouse', 'anteater',
               'MB30418','crocodile', 'terapin','elephant',
               'goat', 'elk', 'lion', 'goldenretriever', ########## animals

			'MB26937', 'MB27203',  'MB27463',
             'MB27876', 'MB27732', 'MB27530', 'MB29650',
             'MB28243', ###### planes
			 
             'face0001', 'face0002', 'face0003',      'face0005', 'face0006',
            'face1', 'face2', 'face3',
            'face5', 'face6', ######## faces
			
            'MB29826', 'MB29342', 'MB28514', 'MB27139',
             'MB27680', 'MB27675', 'MB27692', 'MB27684',  'MB27696',  ##### chairs
			 
			 'MB30374', 'MB30082', 'MB28811', 'MB27386',
             'MB28462', 'MB28077', 'MB28049', 'MB30386',
             'MB30926', 'MB28214', 'antique_furniture_item_17',
             'antique_furniture_item_18', 'antique_furniture_item_20', 'antique_furniture_item_37','antique_furniture_item_44',  ##### tables
			 

				'001M', '002M', '003M', '004M',
               '082M', '087M', '093M', '051M',
               '076M', '054M', '067M', '050M',               '041M', '019M', '020M', '021M', ######### plants
			   
                 'Air_hostess_pose01',
            'Air_hostess_pose02',             'Biochemical_Workman_T',
            'Engineer_pose01',             'Engineer_pose08',
            'Fireman_pose01',                'Fireman_pose08',            'Medic_pose11',
            'Nurse_posing_T',            'Policewoman_pose02',            'Professor_pose04',
            'SWAT_pose06',
            'Soldier_pose02',                'Soldier_pose08',
            'Workman_pose01',               'Workman_pose10', ####### bodies
			]
			 
