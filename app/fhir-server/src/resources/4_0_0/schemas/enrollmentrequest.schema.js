const {
	GraphQLNonNull,
	GraphQLEnumType,
	GraphQLList,
	GraphQLUnionType,
	GraphQLObjectType,
} = require('graphql');
const IdScalar = require('../scalars/id.scalar.js');
const UriScalar = require('../scalars/uri.scalar.js');
const CodeScalar = require('../scalars/code.scalar.js');
const DateTimeScalar = require('../scalars/datetime.scalar.js');

/**
 * @name exports
 * @summary EnrollmentRequest Schema
 */
module.exports = new GraphQLObjectType({
	name: 'EnrollmentRequest',
	description:
		'This resource provides the insurance enrollment details to the insurer regarding a specified coverage.',
	fields: () => ({
		resourceType: {
			type: new GraphQLNonNull(
				new GraphQLEnumType({
					name: 'EnrollmentRequest_Enum_schema',
					values: { EnrollmentRequest: { value: 'EnrollmentRequest' } },
				}),
			),
			description: 'Type of resource',
		},
		_id: {
			type: require('./element.schema.js'),
			description:
				'The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.',
		},
		id: {
			type: IdScalar,
			description:
				'The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.',
		},
		meta: {
			type: require('./meta.schema.js'),
			description:
				'The metadata about the resource. This is content that is maintained by the infrastructure. Changes to the content might not always be associated with version changes to the resource.',
		},
		_implicitRules: {
			type: require('./element.schema.js'),
			description:
				'A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.',
		},
		implicitRules: {
			type: UriScalar,
			description:
				'A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.',
		},
		_language: {
			type: require('./element.schema.js'),
			description: 'The base language in which the resource is written.',
		},
		language: {
			type: CodeScalar,
			description: 'The base language in which the resource is written.',
		},
		text: {
			type: require('./narrative.schema.js'),
			description:
				"A human-readable narrative that contains a summary of the resource and can be used to represent the content of the resource to a human. The narrative need not encode all the structured data, but is required to contain sufficient detail to make it 'clinically safe' for a human to just read the narrative. Resource definitions may define what content should be represented in the narrative to ensure clinical safety.",
		},
		contained: {
			type: new GraphQLList(require('./resourcelist.schema')),
			description:
				'These resources do not have an independent existence apart from the resource that contains them - they cannot be identified independently, and nor can they have their own independent transaction scope.',
		},
		extension: {
			type: new GraphQLList(require('./extension.schema.js')),
			description:
				'May be used to represent additional information that is not part of the basic definition of the resource. To make the use of extensions safe and manageable, there is a strict set of governance  applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.',
		},
		modifierExtension: {
			type: new GraphQLList(require('./extension.schema.js')),
			description:
				"May be used to represent additional information that is not part of the basic definition of the resource and that modifies the understanding of the element that contains it and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and manageable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer is allowed to define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.  Modifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
		},
		identifier: {
			type: new GraphQLList(require('./identifier.schema.js')),
			description: 'The Response business identifier.',
		},
		_status: {
			type: require('./element.schema.js'),
			description: 'The status of the resource instance.',
		},
		status: {
			type: CodeScalar,
			description: 'The status of the resource instance.',
		},
		_created: {
			type: require('./element.schema.js'),
			description: 'The date when this resource was created.',
		},
		created: {
			type: DateTimeScalar,
			description: 'The date when this resource was created.',
		},
		insurer: {
			type: new GraphQLUnionType({
				name: 'EnrollmentRequestinsurer_insurer_Union',
				description: 'The Insurer who is target  of the request.',
				types: () => [require('./organization.schema.js')],
				resolveType(data) {
					if (data && data.resourceType === 'Organization') {
						return require('./organization.schema.js');
					}
				},
			}),
			description: 'The Insurer who is target  of the request.',
		},
		provider: {
			type: new GraphQLUnionType({
				name: 'EnrollmentRequestprovider_provider_Union',
				description:
					'The practitioner who is responsible for the services rendered to the patient.',
				types: () => [
					require('./practitioner.schema.js'),
					require('./practitionerrole.schema.js'),
					require('./organization.schema.js'),
				],
				resolveType(data) {
					if (data && data.resourceType === 'Practitioner') {
						return require('./practitioner.schema.js');
					}
					if (data && data.resourceType === 'PractitionerRole') {
						return require('./practitionerrole.schema.js');
					}
					if (data && data.resourceType === 'Organization') {
						return require('./organization.schema.js');
					}
				},
			}),
			description:
				'The practitioner who is responsible for the services rendered to the patient.',
		},
		candidate: {
			type: new GraphQLUnionType({
				name: 'EnrollmentRequestcandidate_candidate_Union',
				description: 'Patient Resource.',
				types: () => [require('./patient.schema.js')],
				resolveType(data) {
					if (data && data.resourceType === 'Patient') {
						return require('./patient.schema.js');
					}
				},
			}),
			description: 'Patient Resource.',
		},
		coverage: {
			type: new GraphQLUnionType({
				name: 'EnrollmentRequestcoverage_coverage_Union',
				description:
					'Reference to the program or plan identification, underwriter or payor.',
				types: () => [require('./coverage.schema.js')],
				resolveType(data) {
					if (data && data.resourceType === 'Coverage') {
						return require('./coverage.schema.js');
					}
				},
			}),
			description:
				'Reference to the program or plan identification, underwriter or payor.',
		},
	}),
});
