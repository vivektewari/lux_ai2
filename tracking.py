import torch

from vivek_agent import agent_v2
import pandas as pd
class agent_v2(agent_v2):
    def tracking(self, output, key,filename="tracker"):
        """
        save the snapshot of different part of game for reassceing it later
        """
        if key == 'see':
            _, c, l, b = output.shape
            dump = pd.DataFrame(columns=[i for i in range(l)], index=[i for i in range(b)])

            # ['board_size', 'cart', 'cart_COUNT', 'cart_cargo_coal', 'cart_cargo_full', 'cart_cargo_uranium',
            # 'cart_cargo_wood', 'cart_cooldown', 'city_tile', 'city_tile_cooldown', 'city_tile_cost', 'city_tile_fuel',
            # 'coal', 'day_night_cycle', 'dist_from_center_x', 'dist_from_center_y', 'night', 'phase', 'research_points',
            # 'researched_coal', 'researched_uranium', 'road_level', 'turn', 'uranium', 'wood', 'worker', 'worker_COUNT',
            # 'worker_cargo_coal', 'worker_cargo_full', 'worker_cargo_uranium', 'worker_cargo_wood', 'worker_cooldown']
            if self.game_step == 1:
                self.writer = pd.ExcelWriter(self.track_loc +filename+ '.xlsx')
                map = ['board_size', 'cart', 'cart_COUNT', 'cart_cargo_coal', 'cart_cargo_full', 'cart_cargo_uranium',
                       'cart_cargo_wood', 'cart_cooldown', 'city_tile', 'city_tile_cooldown', 'city_tile_cost',
                       'city_tile_fuel',
                       'coal', 'day_night_cycle', 'dist_from_center_x', 'dist_from_center_y', 'night', 'phase',
                       'research_points',
                       'researched_coal', 'researched_uranium', 'road_level', 'turn', 'uranium', 'wood', 'worker',
                       'worker_COUNT',
                       'worker_cargo_coal', 'worker_cargo_full', 'worker_cargo_uranium', 'worker_cargo_wood',
                       'worker_cooldown']
                self.new_map = []
                for i in range(len(map)):
                    self.new_map.append(map[i])
                    if map[i] in ['cart', 'cart_COUNT', 'cart_cargo_coal', 'cart_cargo_full', 'cart_cargo_uranium',
                                  'cart_cargo_wood', 'cart_cooldown', 'city_tile', 'city_tile_cooldown',
                                  'city_tile_cost', 'city_tile_fuel',
                                  'research_points', 'researched_coal', 'researched_uranium', 'worker', 'worker_COUNT',
                                  'worker_cargo_coal', 'worker_cargo_full', 'worker_cargo_uranium', 'worker_cargo_wood',
                                  'worker_cooldown']:
                        self.new_map.append(map[i] + '1')

            next_entry = 'board_size'
            for i in range(l):
                for j in range(b):
                    output_slice = list(output[:, :, i, j].flatten().numpy())

                    dump[i][j] = {}
                    for k in range(c):
                        if output_slice[k] > 0: dump[i][j][self.new_map[k]] = output_slice[k]

            # dump.to_csv(self.track_loc+str(self.game_step)+'_see.csv')

            dump.to_excel(self.writer, sheet_name=str(self.game_step))


            self.writer.save()
        if key == 'act_tracking':
            if self.game_step == 1: self.writer = pd.ExcelWriter(self.track_loc +filename+ '.xlsx')
            if output[3]==0 :return None
            l, b,c = output[1].shape
            dump = pd.DataFrame(columns=[str(i)  for i in range(l)], index=[str(i)  for i in range(b)])

            track_act_list=output[0]
            # for i in range(l):
            #     for j in range(b):
            #         dump[i][j]=""
            for track_act in track_act_list:
                if track_act<19:
                    limit=range(0,19)
                    track_act_=track_act
                elif track_act<36:
                    limit=range(19,36)
                    track_act_=track_act-19
                else :
                    limit=range(36,40)
                    track_act_ = track_act - 36

                old_prob=torch.softmax(output[1].detach()[:,:,limit],dim=2)[:,:,track_act_]
                new_prob=torch.softmax(output[2].detach()[:,:,limit],dim=2)[:,:,track_act_]
                change=new_prob-old_prob
                dump1=pd.DataFrame(new_prob.numpy().transpose())
                dump2=pd.DataFrame(change.numpy().transpose())
                #
                # for i in range(l):
                #     for j in range(b):
                #
                #         change=float(new_prob[i][j])-float(old_prob[i][j])
                #         deck='{},{}'.format(change,float(old_prob[i][j]))#if abs(change)>-0.0000000000000001:
                #         dump[i][j]=deck
                dump1.index.name=str(track_act)
                dump2.index.name = str(track_act)
                dump1.to_excel(self.writer, sheet_name=str(self.game_step)+str("_")+str(output[3]),startrow=track_act_list.index(track_act)*(l+3),startcol=0)
                dump2.to_excel(self.writer, sheet_name=str(self.game_step) + str("_") + str(output[3]),
                               startrow=track_act_list.index(track_act) * (l + 3), startcol=13)
            self.writer.save()
        if key == 'movements':
            if self.game_step == 1: self.writer = pd.ExcelWriter(self.track_loc +filename+ '.xlsx')
            l, b,c = output[1].shape
            dump = pd.DataFrame(columns=[str(i) for i in range(l)], index=[str(i)  for i in range(b)])

            track_act_list=output[0]
            # for i in range(l):
            #     for j in range(b):
            #         dump[i][j]=""
            for track_act in track_act_list:
                if track_act==0:
                    limit=range(0,19)

                elif track_act==1:
                    limit=range(19,36)

                else :
                    limit=range(36,40)



                dump=pd.DataFrame(torch.argmax(torch.softmax(output[2][:,:,limit],dim=2),dim=2).numpy().transpose())
                dump.index.name=str(self.game_step)
                dump.to_excel(self.writer, sheet_name=str(track_act),startrow=(self.game_step-1)*(16),startcol=0)
            # if self.game_step==16:
            #     g=0
            self.writer.save()


