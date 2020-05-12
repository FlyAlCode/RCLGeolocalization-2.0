#include "get_shp_elements.h"
#include <shapefil.h>
#include <string.h>
#include <stdlib.h>


int GetShpElements(const std::string &shp_file_name, 
                   cv::Rect2d &bound, 
                   std::vector<pts> &elements){
    SHPHandle	hSHP;
    int		nShapeType, nEntities, i, iPart, bValidate = 0,nInvalidCount=0;
    int         bHeaderOnly = 0;
    const char 	*pszPlus;
    double 	adfMinBound[4], adfMaxBound[4];
    int nPrecision = 15;
    
    elements.clear();

    /* -------------------------------------------------------------------- */
    /*      Open the passed shapefile.                                      */
    /* -------------------------------------------------------------------- */
    hSHP = SHPOpen( shp_file_name.c_str(), "rb" );

    if( hSHP == NULL )  {
        printf( "Unable to open:%s\n", shp_file_name.c_str() );
        exit( 1 );
    }

    /* -------------------------------------------------------------------- */
    /*      Print out the file bounds.                                      */
    /* -------------------------------------------------------------------- */
    SHPGetInfo( hSHP, &nEntities, &nShapeType, adfMinBound, adfMaxBound );

    // printf( "Shapefile Type: %s   # of Shapes: %d\n\n",
    //         SHPTypeName( nShapeType ), nEntities );
    
    bound.x = adfMinBound[0];
    bound.y = adfMinBound[1];
    bound.width = adfMaxBound[0] - adfMinBound[0];
    bound.height = adfMaxBound[1] - adfMinBound[1];
    
    
    // printf( "File Bounds: (%.*g,%.*g,%.*g,%.*g)\n"
    //         "         to  (%.*g,%.*g,%.*g,%.*g)\n",
    //         nPrecision, adfMinBound[0], 
    //         nPrecision, adfMinBound[1], 
    //         nPrecision, adfMinBound[2], 
    //         nPrecision, adfMinBound[3], 
    //         nPrecision, adfMaxBound[0], 
    //         nPrecision, adfMaxBound[1], 
    //         nPrecision, adfMaxBound[2], 
    //         nPrecision, adfMaxBound[3] );

    /* -------------------------------------------------------------------- */
    /*	Skim over the list of shapes, printing all the vertices.            */
    /* -------------------------------------------------------------------- */
    for( i = 0; i < nEntities; i++ )
    {
        int		j;
        SHPObject	*psShape;

        psShape = SHPReadObject( hSHP, i );

        if( psShape == NULL ) {
            fprintf( stderr,
                     "Unable to read shape %d, terminating object reading.\n",
                    i );
            break;
        }
        
        /* current unused
        if( psShape->bMeasureIsUsed )
            printf( "\nShape:%d (%s)  nVertices=%d, nParts=%d\n"
                    "  Bounds:(%.*g,%.*g, %.*g, %.*g)\n"
                    "      to (%.*g,%.*g, %.*g, %.*g)\n",
                    i, SHPTypeName(psShape->nSHPType),
                    psShape->nVertices, psShape->nParts,
                    nPrecision, psShape->dfXMin,
                    nPrecision, psShape->dfYMin,
                    nPrecision, psShape->dfZMin,
                    nPrecision, psShape->dfMMin,
                    nPrecision, psShape->dfXMax,
                    nPrecision, psShape->dfYMax,
                    nPrecision, psShape->dfZMax,
                    nPrecision, psShape->dfMMax );
        else
            printf( "\nShape:%d (%s)  nVertices=%d, nParts=%d\n"
                    "  Bounds:(%.*g,%.*g, %.*g)\n"
                    "      to (%.*g,%.*g, %.*g)\n",
                    i, SHPTypeName(psShape->nSHPType),
                    psShape->nVertices, psShape->nParts,
                    nPrecision, psShape->dfXMin,
                    nPrecision, psShape->dfYMin,
                    nPrecision, psShape->dfZMin,
                    nPrecision, psShape->dfXMax,
                    nPrecision, psShape->dfYMax,
                    nPrecision, psShape->dfZMax );
        */
        
        if(psShape->nSHPType ==3||psShape->nSHPType ==13||psShape->nSHPType ==23) {     // current we only deal with polyLine
            if( psShape->nParts > 0 && psShape->panPartStart[0] != 0 ) {
                fprintf( stderr, "panPartStart[0] = %d, not zero as expected.\n",
                         psShape->panPartStart[0] );
            }
            
            pts part_pt_tmp;
            for( j = 0, iPart = 1; j < psShape->nVertices; j++ ) {
                if( iPart < psShape->nParts && psShape->panPartStart[iPart] == j ) {                // start a new parts, a part for a element
                        elements.push_back(part_pt_tmp);
                        part_pt_tmp.clear();
                        iPart++;
                }
                part_pt_tmp.push_back(cv::Point2d(psShape->padfX[j], psShape->padfY[j]));
            }
            elements.push_back(part_pt_tmp);
        }
 
        SHPDestroyObject( psShape );
    }

    SHPClose( hSHP );

    
#ifdef USE_DBMALLOC
    malloc_dump(2);
#endif  
}

double GetShpArea(const cv::Rect2d &bound){
    return bound.area();
}

double GetShpTotalLineLength(const std::vector<pts> &elements){
    double total_len = 0;
    for(int i=0; i<elements.size(); i++){
        for(int j=0; j<elements[i].size()-1; j++){
            total_len += cv::norm(elements[i][j+1]-elements[i][j]);
        }
    }
    return total_len;
}

void GetShpInfo(const std::string &shp_file_name, double &area_size, double &total_line_length){
    cv::Rect2d bound; 
    std::vector<pts> elements;
    GetShpElements(shp_file_name, bound, elements);

    area_size = GetShpArea(bound);
    total_line_length = GetShpTotalLineLength(elements);
}


