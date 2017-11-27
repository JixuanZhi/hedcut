#include "wcvt_gpu.h"

std::vector<VorCell> CVT::cells;

float rotateY;
float translateZ;
cv::Mat input_image, grayscale, root;
int iteration = 0;
float max_dist_moved = FLT_MAX;

using namespace std;


void idle_GPU(void)
{
	glutPostRedisplay();
}
void keyboard_GPU(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 'r':
			rotateY += 10.0;
			if (rotateY > 360 || rotateY < -360) rotateY = 0.0;
		break;
		translateZ += 1.0;
		if (translateZ > 1.0) translateZ = 0.0;
		default:
		break;
	}
}
float getDepthValue(int x, int y)
{
	if (x > input_image.size().width || y > input_image.size().height) std::cout << "Access violation with depth buffer" << std::endl;
	float depth = 0.0f;

	glReadBuffer(GL_FRONT);
	glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
	return depth;
}





void CVT::compute_weighted_cvt_GPU(cv::Mat &  img, std::vector<cv::Point2d> & sites)
{
	//init 
	int site_size = sites.size();
	cells.resize(site_size);
	for (int i = 0; i < site_size; i++)
	{
		cells[i].site = sites[i];
	}

	float max_dist_moved = FLT_MAX;

	run_GPU(argc_GPU, argv_GPU, img);
}

void CVT::run_GPU(int argc, char**argv, cv::Mat& img)
{
	//Init opengl
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(img.size().width, img.size().height);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("Image");

	init_GPU(img);

	glutDisplayFunc(display_GPU);
	glutKeyboardFunc(keyboard_GPU);
	glutIdleFunc(idle_GPU);

	glutMainLoop();
}

float CVT::move_site_GPU(VorCell & cell)//move single site
{
    float total = 0;
    cv::Point2d new_pos(0, 0);
    for (auto& c : cell.coverage)
    {
        float d = (256 - input_image.at<uchar>(c.x, c.y))*1.0f / 256;;
        new_pos.x += d*c.x;
        new_pos.y += d*c.y;
        total += d;
    }
    
    //normalize
    new_pos.x /= total;
    new_pos.y /= total;
    
    
    //update
    float dist = fabs(new_pos.x - cell.site.x) + fabs(new_pos.y - cell.site.y); //manhattan dist
    cell.site = new_pos;
    
    //done
    return dist;
}




float CVT::move_sites_GPU() // move all the sites
{
	float max_offset = 0;

    for (auto& cell : cells)
    {
        //cout << "coverage size=" << cvt.cells[607].coverage.size() << endl;
        
        if(cell.coverage.size()!=1)  //keep the empty size
        {
        float offset = move_site_GPU(cell);
        if (offset > max_offset)
        max_offset = offset;
        }
    }
    return max_offset;
}

//buil the VOR once
void CVT::vor_GPU()
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glRotatef(rotateY, 0.0, 1.0, 0.0);

	//Cone information
	GLdouble base = 50.0 ;		//*****
	GLdouble height = 1.0;	//*****
	GLint slices = 50;		//*****
	GLint stacks = 50;

	//Image information
	GLfloat d = 0.0;
	unsigned int r = 0, g = 0, b = 0;
	//GLfloat red = 0.0, green = 0.0, blue = 0.0;
    GLubyte red=0,green=0, blue =0;
    
	//Draw discrete voronoi diagram
	for (int i = 0; i < cells.size(); i++)
	{
        cells[i].coverage.clear();
        
		cv::Point pix(cells[i].site.x, cells[i].site.y);
		root.at<ushort>(pix.x, pix.y) = i;
        
        
		d = (256 - (float)grayscale.at<uchar>(pix.x, pix.y))*1.0f / 256;
		
		/*r = input_image.at<cv::Vec3b>(pix.x, pix.y)[2];
		g = input_image.at<cv::Vec3b>(pix.x, pix.y)[1];
		b = input_image.at<cv::Vec3b>(pix.x, pix.y)[0];*/
        
        // encode the color and id
		if(i/256==0)
        {
        red = (unsigned int)i;
        green = (unsigned int)0;
        blue = (unsigned int)i;
        }
        
        else{
            int a = i%256;
            int b = i/256;
            red = (unsigned int)a;
            green = (unsigned int )b;
            blue = (unsigned int )a;
        }


		
		glPushMatrix();
		//Convert opengl coordinates to opencv coordinates
		glScalef(2.0 / (float)input_image.size().width, 2.0 / (float)input_image.size().height, 1.0);
		glTranslatef(-(float)input_image.size().width / 2.0, (float)input_image.size().height / 2.0, 0.0);
		glRotatef(180.0, 1.0, 0.0, 0.0);

		glTranslatef(cells[i].site.y*1.0f, cells[i].site.x*1.0f, -d);
		glColor3ub(red, green, blue);
		glutSolidCone(base, height, slices, stacks);
		glPopMatrix();
	}
    

   
    
    cv::Mat img;
    img.create(input_image.size().height, input_image.size().width, CV_8UC3);
    

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, img.cols, img.rows, GL_RGB, GL_UNSIGNED_BYTE, img.data);
    cv::Mat flipped;
    cv::flip(img, flipped, 0);
    
   /* cv::imshow("dd",flipped);
    
    if (iteration==0 || iteration==99)
    {
        cv::imwrite("gpu.png",flipped);
    }
    */
   
    //unicode the color and collect the pixel
    for (int x = 0; x < input_image.size().height; x++)
     {
         for (int y = 0; y < input_image.size().width; y++)
         {
             r = flipped.at<cv::Vec3b>(x, y)[2];
             g = flipped.at<cv::Vec3b>(x, y)[1];
             b = flipped.at<cv::Vec3b>(x, y)[0];
             
             ushort rootid = (ushort)r+g*256;
             cells[rootid].coverage.push_back(cv::Point(x,y));
         }//end y
     }//end x*/
    
    // keep the empty site.
    int cvt_size = cells.size();
    for (int i = 0; i < cvt_size; i++)
    {
        if (cells[i].coverage.empty())
        {
            //cells[i] = cells.back();
            //cells.pop_back();
            //i--;
            //cvt_size--;
            cells[i].coverage.push_back(cv::Point(cells[i].site.x,cells[i].site.y));
        }
    }//end for i
    


}


void CVT::init_GPU(cv::Mat& img)
{
	glEnable(GL_DEPTH_TEST);

	//Initialize global variables
	rotateY = 0.0;
	translateZ = 0.0;
	input_image = img;
	cv::cvtColor(input_image, grayscale, CV_BGR2GRAY);
	root = cv::Mat(img.size(), CV_16U, cv::Scalar::all(USHRT_MAX)).clone();

	
}
void CVT::display_GPU(void)
{
	vor_GPU();
   if(max_dist_moved>1 && iteration < 100)
      {
          
      max_dist_moved = move_sites_GPU();
          //cout<<max_dist_moved<<endl;
        iteration++;
    }
    glutSwapBuffers();
}

