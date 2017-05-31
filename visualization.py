import sys
import time
from PyQt4 import QtGui, QtCore

class QTVisualizer(QtGui.QWidget):
    def __init__(self, gamename):
        QtGui.QWidget.__init__(self)

        self.setGeometry(300, 300, 640, 480)
        self.setWindowTitle(gamename)

        import pyqtgraph as pg
        pg.setConfigOption('background', 'w')

        self.plot = pg.PlotWidget()
        self.layout = QtGui.QGridLayout()

        self.setLayout(self.layout)

        self.layout.addWidget(self.plot, 1, 0, 3, 1)
        self.show()

        self.obs_brush = pg.mkBrush(cosmetic=False, width=4.5, color='k')
        self.agent_brush = pg.mkBrush(cosmetic=False, width=4.5, color='r')

        self.goal_brush = pg.mkBrush(cosmetic=False, width=4.5, color='g')

	self.w = 12
	self.h = 12

        self.plot.setRange(xRange=[0,self.w-1], yRange=[0,self.h-1])

        self.initialized = False

    def draw_world(self, agent, obstacles, goals):

        w = self.w
        h = self.h

        if not self.initialized:

            self.agent_plotitem = self.plot.plot(x = [agent[0]], y=[agent[1]], symbolBrush=self.agent_brush, symbol='o')

            self.list_of_obs_plotitems = []
            for obs in obstacles:
                obs_plotitem = self.plot.plot(x = [obs[0]], y = [obs[1]], symbolBrush=self.obs_brush, symbol='s')
                self.list_of_obs_plotitems.append(obs_plotitem)

            self.list_of_goal_plotitems = []
            for goal in goals:
                goal_plotitem = self.plot.plot(x = [goal[0]], y=[goal[1]], symbolBrush=self.goal_brush, symbol='s')
                self.list_of_goal_plotitems.append(goal_plotitem)
            
            self.initialized = True
        else:
            self.agent_plotitem.setData(x=[int(round(agent[0]))], y=[int(round(agent[1]))])
            for obs, obs_plotitem in zip(obstacles, self.list_of_obs_plotitems):
                obs_plotitem.setData(x=[int(round(obs[0]))], y=[int(round(obs[1]))])

            for goal, goal_plotitem in zip(goals, self.list_of_goal_plotitems):
                goal_plotitem.setData(x = [int(round(goal[0]))], y=[int(round(goal[1]))])


def q_refresh():
    qApp = QtGui.QApplication.instance()
    qApp.processEvents()
    time.sleep(0.5)


#def main():
#    app = QtGui.QApplication(sys.argv)
#    visualizer = QTVisualizer('Varying rewards')
#    for i in range(10000):
#        visualizer.draw_world(agent=(0,5), obstacles=[(3,3),(3,6),(6,3),(6,6)], goals=[(11,0),(11,11)])
#        q_refresh() 


#if __name__ == '__main__':
#    main()
