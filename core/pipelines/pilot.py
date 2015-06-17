import os,sys,glob,datetime
import numpy as np

from nipype.interfaces import spm, fsl, afni, nitime, utility, lif, dcmstack as np_dcmstack, freesurfer, nipy

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nibabel as nb

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
afni.base.AFNICommand.set_default_output_type('NIFTI_GZ')
np_dcmstack.DCMStackBase.set_default_output_type('NIFTI_GZ')

sys.path.insert(0,'/home/bpinsard/data/src/misc')
import generic_pipelines
from generic_pipelines.utils import wrap, fname_presuffix_basename, wildcard

from nipype import config
cfg = dict(execution={'stop_on_first_crash': False})
config.update_config(cfg)

data_dir = '/home/bpinsard/data/raw/UNF/CoRe'
proc_dir = '/home/bpinsard/data/analysis/'

subjects = ['S40_JB']

tr=2.16
file_pattern = '_%(PatientName)s_%(SeriesDate)s_%(SeriesTime)s'
meta_tag_force=['PatientID','PatientName','SeriesDate']

def dicom_dirs():

    subjects_info = pe.Node(
        utility.IdentityInterface(fields=['subject']),
        name='subjects_info')
    subjects_info.iterables = [('subject', subjects)]

    dicom_dirs = pe.Node(
        nio.DataGrabber(infields=['subject'],
                        outfields=['t1_mprage','aa_scout','localizer',
                                   'fmri_resting_state','fmri_task1','fmri_task2',
                                   'fmri_fieldmap', 'fmri_pa'],
                        sort_filelist = True,
                        base_directory = data_dir, template=''),
        name='dicom_dirs')
    dicom_dirs.inputs.template = 'CoRe_%s_*/??-%s'
    dicom_dirs.inputs.template_args = dict(
        t1_mprage=[['subject','anat_mprage-MPRAGE_12ch']],
        aa_scout=[['subject','AAScout']],
        localizer=[['subject','localizer_12Channel']],
        fmri_resting_state=[['subject','fmri-BOLD_Resting_State']],
        fmri_task1=[['subject','fmri-BOLD_Task1']],
        fmri_task2=[['subject','fmri-BOLD_Task2']],
        fmri_fieldmap=[['subject','fmri_fieldmap-gre_field_map_BOLD/*']],
        fmri_pa=[['subject','fmri-BOLD_PA']],)

    w=pe.Workflow(name='core_pilot')

    for n in [dicom_dirs]:
        w.connect([(subjects_info,n,[('subject',)*2])])
        
    return w
    
def preproc_anat():
    
    w = dicom_dirs()

    n_t1_convert = pe.Node(
        np_dcmstack.DCMStackAnatomical(
            meta_force_add=meta_tag_force,
            out_file_format = 't1_mprage'+file_pattern,
            voxel_order='LAS'),
        name='convert_t1_dicom')

    n_fieldmap_fmri_convert = pe.Node(
        np_dcmstack.DCMStackFieldmap(
            meta_force_add=meta_tag_force,
            out_file_format = 'fieldmap_bold'+file_pattern,
            voxel_order='LAS'),
        name='convert_fieldmap_dicom')

    t1_pipeline = generic_pipelines.t1_new.t1_freesurfer_pipeline()
    t1_pipeline.inputs.freesurfer.args=''
    t1_pipeline.inputs.freesurfer.openmp = 8
    wm_surface = generic_pipelines.t1_new.extract_wm_surface()

    n_crop_t1 = pe.Node(
        nipy.Crop(out_file='%s_crop.nii.gz',
                  outputtype='NIFTI_GZ'),
        name='crop_t1')
    t1_pipeline.connect([
            (t1_pipeline.get_node('autobox_mask_fs'),n_crop_t1,
             [('%s_%s'%(c,p),)*2 for c in 'xyz' for p in ('min','max')]),
            (t1_pipeline.get_node('freesurfer'),n_crop_t1,
             [('norm','in_file')])])

    n_flirt_fmap2t1 = pe.Node(
        fsl.FLIRT(cost='mutualinfo',
                  dof=6,
                  out_matrix_file = 'fmap2t1.mat',
                  no_search=True,
                  uses_qform=True),
        name='flirt_fmap2t1')

    n_flirt2xfm = pe.Node(
        freesurfer.preprocess.Tkregister(reg_file='reg.mat',
                                         xfm_out='fmap2t1.xfm',
                                         freeview='freeview.mat',
                                         fsl_reg_out='fsl_reg_out.mat',
                                         lta_out='lta.mat',),
        name='flirt2xfm')

    n_xfm2mat = pe.Node(
        utility.Function(
            input_names = ['xfm'],
            output_names = ['mat'],
            function='def f(xfm): import numpy as np; mat=np.loadtxt(xfm, skiprows=5,usecols=range(4)); np.savetxt("out.mat",np.vstack([mat,[0,0,0,1]]));return "out.mat"'),
        name='xfm2mat')

    n_fs32k_surf = generic_pipelines.fmri_surface.surface_32k()

    n_low_res_parc = pe.Node(
        freesurfer.MRIConvert(vox_size=(2.,)*3,out_type='niigz',
                              args='-rt nearest'),
        name='low_res_parc')
    n_reg_crop = pe.Node(
        freesurfer.Tkregister(reg_file='reg_crop.dat',
                              freeview='freeview.mat',
                              fsl_reg_out='fsl_reg_out.mat',
                              lta_out='lta.mat',
                              xfm_out='xfm.mat',
                              reg_header=True),
        'reg_crop')
    n_compute_pvmaps = pe.Node(
        freesurfer.ComputeVolumeFractions(),
        name='compute_pvmaps')

    t1_pipeline.connect([
            (t1_pipeline.get_node('freesurfer'),n_reg_crop,[
                    ('norm','target'),('subject_id',)*2,('subjects_dir',)*2]),
            (t1_pipeline.get_node('autobox_mask_fs'),n_reg_crop,[
                    ('out_file','mov')]),
            (n_reg_crop, n_compute_pvmaps, [
                    ('reg_file',)*2]),
            (t1_pipeline.get_node('autobox_mask_fs'), n_compute_pvmaps, [
                    ('out_file','in_file')]),
            (t1_pipeline.get_node('freesurfer'),n_compute_pvmaps,[
                    ('subjects_dir',)*2]),            
            ])
    
    w.base_dir = proc_dir
    w.connect([
            (w.get_node('dicom_dirs'),n_t1_convert,[('t1_mprage','dicom_files')]),
            (w.get_node('subjects_info'),t1_pipeline,[('subject','inputspec.subject_id')]),
            (n_t1_convert,t1_pipeline,[('nifti_file','inputspec.t1_file')]),
            (t1_pipeline, wm_surface, [(('freesurfer.aparc_aseg',utility.select,0),'inputspec.aseg')]),
            (t1_pipeline,n_fs32k_surf,[('freesurfer.subjects_dir','fs_source.base_directory'),]),
            (w.get_node('subjects_info'),n_fs32k_surf,[('subject','fs_source.subject')]),
            (w.get_node('dicom_dirs'),n_fieldmap_fmri_convert,[('fmri_fieldmap', 'dicom_files')]),

            (n_fieldmap_fmri_convert, n_flirt_fmap2t1,[('magnitude_file','in_file')]),
            (t1_pipeline, n_flirt_fmap2t1,[('crop_t1.out_file','reference'),]),
            
            (n_flirt_fmap2t1, n_flirt2xfm,[('out_matrix_file','fsl_reg')]),
            (n_flirt2xfm, n_xfm2mat,[('xfm_out','xfm')]),
            (n_fieldmap_fmri_convert, n_flirt2xfm,[('magnitude_file','mov')]),
            (t1_pipeline, n_flirt2xfm,[('crop_t1.out_file','target'),]),

       ])

    return w



def preproc_fmri():
    
    w = dicom_dirs()
    templates = dict(
        subjects_dir=[['t1_preproc/_subject','subject','freesurfer']],
        norm = [['t1_preproc/_subject','subject','freesurfer/*[!e]/mri/norm.mgz']],
        white_matter_surface = [['extract_wm_surface/_subject','subject','surf_decimate/rlh.aparc+aseg_wm.nii_smoothed_mask.all']],
        cropped_mask = [['t1_preproc/_subject','subject','autobox_mask_fs/*.nii.gz']],
        cropped_t1 = [['t1_preproc/_subject','subject','crop_t1/*.nii.gz']],
        pve_maps = [
            ['t1_preproc/_subject','subject','compute_pvmaps/*.gm.mgz'],
            ['t1_preproc/_subject','subject','compute_pvmaps/*.wm.mgz'],
            ['t1_preproc/_subject','subject','compute_pvmaps/*.csf.mgz']],
        fieldmaps = [['_subject','subject','convert_fieldmap_dicom/*field.nii.gz']],
        fieldmap_regs = [['_subject','subject','xfm2mat/out.mat']],
        lowres_surf_lh = [
            ['surface_32k/_subject','subject',
             'white_resample_surf/mapflow/_white_resample_surf0/lh.white_converted.32k.gii'],
            ['surface_32k/_subject','subject',
             'pial_resample_surf/mapflow/_pial_resample_surf0/lh.pial_converted.32k.gii']],
        lowres_surf_rh = [
            ['surface_32k/_subject','subject',
             'white_resample_surf/mapflow/_white_resample_surf1/rh.white_converted.32k.gii'],
            ['surface_32k/_subject','subject',
             'pial_resample_surf/mapflow/_pial_resample_surf1/rh.pial_converted.32k.gii']],
        lowres_rois = [['t1_preproc/_subject','subject','low_res_parc/aparc.a2009s+aseg_out.nii.gz']])
    n_anat_grabber = pe.Node(
        nio.DataGrabber(
            infields=['subject'],
            outfields=templates.keys(),
            sort_filelist=True, raise_on_empty=False,
            base_directory = proc_dir),
        name='anat_grabber')
    n_anat_grabber.inputs.template = 'core_pilot/%s_%s/%s'
    n_anat_grabber.inputs.template_args = templates

    def name_struct(f,n,*args):
        if isinstance(f,list):
            f = tuple(f)
        return [tuple([n,f]+list(args))]

    def repeater(l,n):
        return [l]*n

    n_group_hemi_surfs = pe.Node(
        utility.Merge(2),
        name='group_hemi_surfs')

    fmri_proc = generic_pipelines.fmri_surface.fmri_surface_preproc()
    fmri_proc.inputs.inputspec.echo_spacing = .000265
    fmri_proc.inputs.inputspec.repetition_time  = tr
    fmri_proc.inputs.inputspec.echo_time = .03
    fmri_proc.inputs.inputspec.phase_encoding_dir = 1

    n_merge_fmri_sequences = pe.Node(
        utility.Merge(3),
        name='merge_fmri_sequences')

    n_fmri_convert = pe.MapNode(
        np_dcmstack.DCMStackfMRI(
            meta_force_add=meta_tag_force,
            out_file_format = 'fmri'+file_pattern),
        iterfield=['dicom_files'],
        name='fmri_convert')

    n_st_realign = pe.Node(
        nipy.FmriRealign4d(
            tr=tr,
            slice_order=range(40),
            time_interp=True),
        name='st_realign')

    n_bbreg_epi = pe.Node(
        freesurfer.BBRegister(init='header',contrast_type='t2',reg_frame=0),
        name='bbreg_epi')

    n_mask2epi= pe.Node(
        freesurfer.ApplyVolTransform(interp='nearest',inverse=True),
        name='mask2epi')
    """
    n_bbreg2xfm = pe.Node(
        freesurfer.preprocess.Tkregister(xfm_out='fmap2t1.xfm',
                                         freeview='freeview.mat',
                                         fsl_reg_out='fsl_reg_out.mat',
                                         lta_out='lta.mat',),
        name='bbreg2xfm')
        """
    n_convert_motion_par = pe.MapNode(
        utility.Function(
            input_names = ['motion','epi2t1','matrix_file'],
            output_names = ['motion'],
            function='def f(motion,epi2t1,matrix_file): from os.path import abspath;import numpy as np; import nibabel as nb ;from nipy.algorithms.registration.affine import to_matrix44;mot=np.loadtxt(motion); epi2t1reg=np.loadtxt(epi2t1); reg=nb.load(matrix_file).affine; mats = np.array([epi2t1reg.dot(np.linalg.inv(to_matrix44(m)).dot(reg)) for m in mot for i in range(40)]); out_fname=abspath("motion.npy"); np.save(out_fname,mats);return out_fname'),
        iterfield=['motion','matrix_file'],
        name='convert_motion_par')
    n_convert_motion_par.inputs.epi2t1='/home/bpinsard/data/analysis/core_pilot/_subject_S40_JB/out.xfm'

    n_noise_corr = pe.MapNode(
        nipy.preprocess.OnlineFilter(
            echo_time=.03,
            echo_spacing=.000265,
            phase_encoding_dir=1,
            resampled_first_frame='frame1.nii',
            out_file_format='ts.h5'),
        iterfield = ['dicom_files','motion'],
        overwrite=False,
        name = 'noise_correction')

    n_smooth_bp = pe.MapNode(
        generic_pipelines.fmri_surface.GrayOrdinatesBandPassSmooth(
            data_field = 'FMRI/DATA',
            smoothing_factor=3,
            smoothing_steps=3,
            filter_range=(.008,.1),
            TR=2.16),
        iterfield = ['in_file'],
        name = 'smooth_bp')

    bold_seqs = ['fmri_resting_state','fmri_task1','fmri_task2']

    w.base_dir = proc_dir
    si = w.get_node('subjects_info')
    w.connect([
            (si, n_anat_grabber, [('subject',)*2]),
            (w.get_node('dicom_dirs'),n_merge_fmri_sequences,[(seq,'in%d'%(i+1)) for i,seq in enumerate(bold_seqs)]),
#            (n_anat_grabber, fmri_proc,[
#                    ('white_matter_surface','inputspec.reference_boundary'),
#                    ('norm','inputspec.surfaces_volume_reference'),
#                    ('cropped_mask','inputspec.mask'),
#                    (('lowres_rois',name_struct,'SUBCORTICAL_CEREBELLUM',
#                      '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
#                     'inputspec.resample_rois'),
#                    (('fieldmaps',repeater,3),'inputspec.fieldmap'),
#                    (('fieldmap_regs',repeater,3),'inputspec.fieldmap_reg'),
#                    (('fieldmap_regs',repeater,3),'inputspec.init_reg')
#                    ]),

             (n_anat_grabber, n_group_hemi_surfs,[
                    (('lowres_surf_lh',name_struct,'CORTEX_LEFT'),'in1'),
                    (('lowres_surf_rh',name_struct,'CORTEX_RIGHT'),'in2')]),
#            (n_group_hemi_surfs, fmri_proc,[('out','inputspec.resample_surfaces')]),

#            (n_merge_fmri_sequences,fmri_proc,[('out','inputspec.dicom_files')]),
            (n_merge_fmri_sequences,n_fmri_convert,[('out','dicom_files')]),
            (n_fmri_convert,n_st_realign,[('nifti_file','in_file')]),
            (n_anat_grabber,n_bbreg_epi,[('subjects_dir',)*2]),
            (si,n_bbreg_epi,[('subject','subject_id')]),
            (n_st_realign,n_bbreg_epi,[(('out_file',generic_pipelines.utils.getitem_rec,0),'source_file')]),
#            (n_st_realign,n_bbreg2xfm,[(('out_file',generic_pipelines.utils.getitem_rec,0),'mov')]),
#            (n_bbreg_epi, n_bbreg2xfm,[('out_reg_file','reg_file')]),
#            (n_anat_grabber,n_bbreg2xfm,[('subjects_dir',)*2]),            

#            (n_bbreg_epi,n_convert_motion_par,[('out_reg_file','epi2t1')]),
            (n_st_realign, n_convert_motion_par,[('par_file','motion')]),
            (n_fmri_convert, n_convert_motion_par,[('nifti_file','matrix_file')]),
            (n_convert_motion_par, n_noise_corr,[('motion','motion')]),
            (n_merge_fmri_sequences,n_noise_corr,[('out','dicom_files')]),
            (n_anat_grabber, n_noise_corr,[
                    ('norm','surfaces_volume_reference'),
                    ('cropped_mask','mask'),
                    (('lowres_rois',name_struct,'SUBCORTICAL_CEREBELLUM',
                      '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
                     'resample_rois'),
                    ('fieldmaps','fieldmap'),
                    ('fieldmap_regs','fieldmap_reg'),
                    ('pve_maps','partial_volume_maps')
                    ]),
            (n_group_hemi_surfs, n_noise_corr,[('out','resample_surfaces')]),
            (n_noise_corr, n_smooth_bp,[('out_file','in_file')]),
            
            (n_anat_grabber, fmri_proc, [('pve_maps','inputspec.partial_volume_maps')]),
            ])
    

    return w 


from mvpa2.clfs.meta import BinaryClassifier, CombinedClassifier, MaximalVote, MappedClassifier
from mvpa2.generators.splitters import Splitter
from sklearn.svm import OneClassSVM
from mvpa2.clfs.skl import SKLLearnerAdapter

def get_classifier():
    

    clf = SKLLearnerAdapter(OneClassSVM(kernel='linear'))
    seqA_clf = BinaryClassifier(clf.clone(),['seqA'],['rest', 'rest_block_A', 'rest_block_B', 'seqB'])
    seqB_clf = BinaryClassifier(clf.clone(),['seqB'],['rest', 'rest_block_A', 'rest_block_B', 'seqA'])
    map_seqA_clf = MappedClassifier(seqA_clf,Splitter(attr='targets',attr_values=[1]))
    map_seqB_clf = MappedClassifier(seqB_clf,Splitter(attr='targets',attr_values=[1]))

    comb_clf = CombinedClassifier([map_seqA_clf,map_seqB_clf], combiner=MaximalVote())
    
    
